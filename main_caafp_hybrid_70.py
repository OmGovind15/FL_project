#
# FILENAME: main_caafp_hybrid.py
#
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import sys
import copy
from datetime import datetime

from data_loader import load_wisdm_dataset, create_non_iid_data_split, create_tf_datasets
from models import create_lstm_model, get_model_sparsity

# --- (Logger Class - unchanged from final_2.py) ---
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message); self.logfile.write(message); self.flush()
    def flush(self):
        self.terminal.flush(); self.logfile.flush()
    def __del__(self):
        sys.stdout = self.terminal
        if self.logfile: self.logfile.close()

# --- (ClusterAwareImportanceScorer - unchanged from final_2.py) ---
# This is the "Hybrid" score (Magnitude + Coherence + Consistency)
class ClusterAwareImportanceScorer:
    def __init__(self, alpha=0.5, beta=0.25, gamma=0.25):
        self.alpha = alpha; self.beta = beta; self.gamma = gamma

    def calculate_scores(self, dense_cluster_model, client_models_in_cluster, client_gradients_in_cluster):
        importance_scores = {}
        for layer in dense_cluster_model.layers:
            if not hasattr(layer, 'kernel'): continue
            layer_name = layer.name
            cluster_weights = layer.kernel.numpy()
            
            # 1. Magnitude
            magnitude_scores = np.abs(cluster_weights)
            mag_max = np.max(magnitude_scores); magnitude_scores = magnitude_scores / mag_max if mag_max > 0 else magnitude_scores
            
            # 2. Coherence
            client_weights_list = []
            for client_model in client_models_in_cluster:
                for client_layer in client_model.layers:
                    if client_layer.name == layer_name and hasattr(client_layer, 'kernel'):
                        client_weights_list.append(client_layer.kernel.numpy()); break
            if client_weights_list:
                variance = np.var(np.array(client_weights_list), axis=0)
                coherence_scores = 1.0 / (1.0 + variance)
                coh_max = np.max(coherence_scores); coherence_scores = coherence_scores / coh_max if coh_max > 0 else coherence_scores
            else:
                coherence_scores = np.ones_like(cluster_weights)
            
            # 3. Consistency
            client_grads_list = []
            for client_grads_dict in client_gradients_in_cluster:
                if layer_name in client_grads_dict:
                    client_grads_list.append(client_grads_dict[layer_name])
            if client_grads_list:
                client_signs = np.sign(np.array(client_grads_list))
                consistency_scores = np.abs(np.mean(client_signs, axis=0))
            else:
                consistency_scores = np.ones_like(cluster_weights)
                
            # Combine
            hybrid_score = (self.alpha * magnitude_scores + 
                          self.beta * coherence_scores + 
                          self.gamma * consistency_scores)
            importance_scores[layer_name] = hybrid_score
        return importance_scores

# --- (AdaptivePruningScheduler - unchanged from final_2.py) ---
class AdaptivePruningScheduler:
    def __init__(self, base_sparsity=0.7, max_sparsity=0.9, min_sparsity=0.5):
        self.base_sparsity = base_sparsity; self.max_sparsity = max_sparsity; self.min_sparsity = min_sparsity
    
    def get_sparsity_for_cluster(self, client_entropy_scores, num_clients_in_cluster):
        # --- FIXED 70% SPARSITY SHOWDOWN ---
        return 0.7

# --- MODIFIED ---
class CAAFPServer:
    """
    Server for Progressive & Dynamic CA-AFP.
    - Stores evolving masks.
    - Implements prune-and-grow logic.
    """
    def __init__(self, num_clients, num_clusters=3):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.global_model = create_lstm_model()
        self.cluster_models = {}
        self.client_clusters = {}
        self.clustered = False
        
        # --- NEW: Store evolving masks ---
        self.cluster_masks = {}
        # --- NEW: Store target sparsities ---
        self.cluster_target_sparsity = {}

        # Use the original Hybrid Scorer
        self.importance_scorer = ClusterAwareImportanceScorer(
            alpha=0.5, beta=0.25, gamma=0.25
        )
        self.pruning_scheduler = AdaptivePruningScheduler(
            base_sparsity=0.7, max_sparsity=0.9, min_sparsity=0.5
        )
    
    # --- (aggregate_weights - unchanged) ---
    def aggregate_weights(self, client_weights):
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        for weights in client_weights:
            for i, w in enumerate(weights): avg_weights[i] += w
        for i in range(len(avg_weights)): avg_weights[i] /= len(client_weights)
        return avg_weights
    
    # --- (cluster_clients - MODIFIED) ---
    def cluster_clients(self, client_updates):
        # (Clustering logic is unchanged)
        flattened_updates = [np.concatenate([u.flatten() for u in update]) for update in client_updates]
        flattened_updates = np.array(flattened_updates)
        n_clients = len(flattened_updates)
        similarity_matrix = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(n_clients):
                similarity_matrix[i, j] = 1 - cosine(flattened_updates[i], flattened_updates[j]) if i != j else 1.0
        distance_matrix = 1 - similarity_matrix
        try:
            clustering = AgglomerativeClustering(n_clusters=self.num_clusters, metric='precomputed', linkage='average')
        except TypeError:
            clustering = AgglomerativeClustering(n_clusters=self.num_clusters, affinity='precomputed', linkage='average')
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        for client_id, cluster_id in enumerate(cluster_labels):
            self.client_clusters[client_id] = int(cluster_id)
        
        # Initialize models and masks
        for cluster_id in range(self.num_clusters):
            self.cluster_models[cluster_id] = create_lstm_model()
            self.cluster_models[cluster_id].set_weights(self.global_model.get_weights())
            
            # --- NEW: Initialize empty mask for each layer ---
            self.cluster_masks[cluster_id] = {}
            for layer in self.cluster_models[cluster_id].layers:
                if hasattr(layer, 'kernel'):
                    self.cluster_masks[cluster_id][layer.name] = np.ones_like(layer.kernel.numpy())
        
        self.clustered = True
        return cluster_labels

    # --- NEW: Function to set the final target sparsity ---
    def set_cluster_sparsity_targets(self, clients):
        print("\n--- PHASE 4: Calculating Adaptive Sparsity Targets ---")
        for cluster_id in range(self.num_clusters):
            cluster_client_ids = [cid for cid, cl in self.client_clusters.items() if cl == cluster_id]
            if not cluster_client_ids:
                self.cluster_target_sparsity[cluster_id] = self.pruning_scheduler.base_sparsity
                continue
                
            client_entropies = [clients[cid].get_local_label_entropy() for cid in cluster_client_ids]
            
            target_sparsity = self.pruning_scheduler.get_sparsity_for_cluster(
                client_entropies, len(cluster_client_ids)
            )
            self.cluster_target_sparsity[cluster_id] = target_sparsity
            print(f"  Cluster {cluster_id}: Target Sparsity = {target_sparsity:.2%}")

    # --- NEW: Progressive & Dynamic Pruning Function --
    def update_masks_and_evolve(self, cluster_id, client_models_in_cluster, client_gradients_in_cluster, current_sparsity, 
                               target_sparsity, round_num, prune_rate=0.05):
        """
        Implements Requirements 1 & 2.
        Progressively prunes and dynamically grows the mask.
        """
        # --- FIX 1: Correct variable in print statement ---
        print(f"  Evolving Mask for Cluster {cluster_id} (Current: {current_sparsity:.2%}, Target: {target_sparsity:.2%})")
        
        dense_model = self.cluster_models[cluster_id]
        
        # 1. Get Hybrid Importance Scores
        importance_scores = self.importance_scorer.calculate_scores(
            dense_model, client_models_in_cluster, client_gradients_in_cluster
        )
        
        # 2. Get Gradient Magnitudes (for growing)
        avg_gradients = {}
        for layer_name in importance_scores.keys():
            grads_list = [c_grads[layer_name] for c_grads in client_gradients_in_cluster if layer_name in c_grads]
            if grads_list:
                avg_gradients[layer_name] = np.mean(np.array(grads_list), axis=0)
            else:
                avg_gradients[layer_name] = np.zeros_like(importance_scores[layer_name])

        # 3. Evolve mask layer by layer
        for layer in dense_model.layers:
            if layer.name not in importance_scores:
                continue
            
            layer_name = layer.name
            current_mask = self.cluster_masks[cluster_id][layer_name]
            
            hybrid_scores = importance_scores[layer_name]
            grad_scores = np.abs(avg_gradients[layer_name])
            
            # --- START FIX 2 & 3: Sparsity Logic + Type Casting ---
            # --- START FINAL FIX: Deficit + Churn Logic ---
            n_total = current_mask.size
            n_active = np.sum(current_mask)
            
            # 1. Calculate sparsity gap
            current_sparsity = (n_total - n_active) / n_total
            sparsity_gap = target_sparsity - current_sparsity
            
            # 2. Calculate remaining PRUNING STEPS (not rounds)
            # Total cluster rounds = 30 (from round 10 to 40)
            # Pruning frequency = 5
            # Total pruning steps = 30 / 5 = 6 steps
            rounds_elapsed = round_num - 10 # 10 is initial rounds
            pruning_steps_elapsed = rounds_elapsed // 5 # How many pruning steps have passed
            pruning_steps_total = 30 // 5
            pruning_steps_remaining = max(1, pruning_steps_total - pruning_steps_elapsed) 
            
            # 3. Calculate "deficit push" (net change needed)
            n_deficit_push = int(sparsity_gap * n_total / pruning_steps_remaining)
            
            # 4. Calculate "dynamic churn" (based on 5% prune_rate of *active* weights)
            n_churn = int(n_active * prune_rate) 
            
            # 5. Determine final prune/grow numbers
            if n_deficit_push > 0:
                # We are UNDER target (need to prune more)
                n_to_prune = n_churn + n_deficit_push
                n_to_grow = n_churn
            else:
                # We are OVER target (need to grow more)
                n_to_prune = n_churn
                n_to_grow = n_churn - n_deficit_push # n_deficit_push is negative, so this adds
            
            # 6. Cast to int() and ensure we don't prune/grow more than available
            n_to_prune = int(min(n_active, max(0, n_to_prune))) # Ensure non-negative
            n_to_grow = int(min(n_total - n_active, max(0, n_to_grow))) # Ensure non-negative
            # --- END FINAL FIX ---
            # --- END FIX ---
            
            flat_mask = current_mask.flatten()
            flat_hybrid_scores = hybrid_scores.flatten()
            flat_grad_scores = grad_scores.flatten()

            # --- PRUNE LOGIC ---
            active_indices = np.where(flat_mask > 0)[0]
            active_scores = flat_hybrid_scores[active_indices]
            indices_to_prune = active_indices[np.argsort(active_scores)[:n_to_prune]] # Slice is now safe
            
            # --- GROW LOGIC ---
            inactive_indices = np.where(flat_mask == 0)[0]
            if inactive_indices.size > 0:
                inactive_grad_scores = flat_grad_scores[inactive_indices]
                indices_to_grow = inactive_indices[np.argsort(inactive_grad_scores)[-n_to_grow:]] # Slice is now safe
            else:
                indices_to_grow = []

            flat_mask[indices_to_prune] = 0.0
            flat_mask[indices_to_grow] = 1.0
            
            self.cluster_masks[cluster_id][layer_name] = flat_mask.reshape(current_mask.shape)

    # --- REMOVED ---
    # The old `apply_cluster_aware_pruning` is gone.
    
    def get_final_pruned_model(self, cluster_id):
        """Applies the final mask to the dense model."""
        dense_model = self.cluster_models[cluster_id]
        pruned_model = create_lstm_model()
        pruned_model.set_weights(dense_model.get_weights())
        
        for layer in pruned_model.layers:
            if layer.name in self.cluster_masks[cluster_id]:
                mask = self.cluster_masks[cluster_id][layer.name]
                weights = layer.kernel.numpy()
                layer.kernel.assign(weights * mask)
        return pruned_model


# --- MODIFIED ---
class CAAFPClient:
    def __init__(self, client_id, train_dataset, test_dataset, lambda_reg=0.01):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lambda_reg = lambda_reg
        self.personal_model = create_lstm_model()
        self.cluster_model = create_lstm_model() # This model will be masked
    
    # --- NEW: Helper function to apply mask ---
    def apply_mask(self, cluster_mask):
        """Applies the cluster mask to the client's cluster_model."""
        for layer in self.cluster_model.layers:
            if layer.name in cluster_mask:
                weights = layer.kernel.numpy()
                mask = cluster_mask[layer.name]
                layer.kernel.assign(weights * mask)

    # --- MODIFIED: To accept and apply the mask ---
    def train_with_regularization(self, cluster_weights, cluster_mask, epochs=5):
        """
        Train with regularization, applying the evolving mask.
        """
        self.cluster_model.set_weights(cluster_weights)
        self.apply_mask(cluster_mask) # Apply mask at the start
        
        for epoch in range(epochs):
            for x, y in self.train_dataset:
                with tf.GradientTape() as tape:
                    # ... (Regularization loss logic is unchanged) ...
                    predictions = self.personal_model(x, training=True)
                    local_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, predictions))
                    reg_loss = 0.0
                    for pw, cw in zip(self.personal_model.trainable_weights,
                                     self.cluster_model.trainable_weights):
                        reg_loss += tf.reduce_sum(tf.square(pw - cw))
                    reg_loss = (self.lambda_reg / 2.0) * reg_loss
                    total_loss = local_loss + reg_loss
                
                gradients = tape.gradient(total_loss, self.personal_model.trainable_weights)
                self.personal_model.optimizer.apply_gradients(
                    zip(gradients, self.personal_model.trainable_weights)
                )
            
            # Train the cluster model (which is masked)
            self.cluster_model.fit(self.train_dataset, epochs=1, verbose=0)
            
            # --- NEW: Re-apply mask after fit() to ensure pruned weights stay zero ---
            self.apply_mask(cluster_mask)
        
        return self.cluster_model.get_weights()

    # --- MODIFIED: Fine-tune the *personal* model, not a new one ---
    def fine_tune(self, pruned_model_weights, epochs=3):
        """Fine-tune the personal model."""
        self.personal_model.set_weights(pruned_model_weights)
        # Apply mask? No, fine-tuning is on the final pruned model.
        # But we must ensure the *personal* model is also pruned.
        
        # Let's clone the structure and weights
        self.personal_model = create_lstm_model()
        self.personal_model.set_weights(pruned_model_weights)
        
        self.personal_model.fit(self.train_dataset, epochs=epochs, verbose=0)

    # --- (compute_gradients - unchanged) ---
    def compute_gradients(self, model_weights):
        self.cluster_model.set_weights(model_weights)
        gradients = {}
        for x, y in self.train_dataset.take(1):
            with tf.GradientTape() as tape:
                predictions = self.cluster_model(x, training=False)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, predictions))
            trainable_weights_map = {var.name: var for var in self.cluster_model.trainable_weights}
            grads = tape.gradient(loss, list(trainable_weights_map.values()))
            grad_map = {var.name: grad for var, grad in zip(trainable_weights_map.values(), grads)}
            for layer in self.cluster_model.layers:
                if hasattr(layer, 'kernel'):
                    kernel_name = f"{layer.name}/kernel:0"
                    if kernel_name in grad_map and grad_map[kernel_name] is not None:
                        gradients[layer.name] = grad_map[kernel_name].numpy()
        return gradients
    
    # --- (compute_update, evaluate, get_local_label_entropy - unchanged) ---
    def compute_update(self, old_weights):
        new_weights = self.cluster_model.get_weights(); return [new - old for new, old in zip(new_weights, old_weights)]
    def evaluate(self):
        results = self.personal_model.evaluate(self.test_dataset, verbose=0); return {'loss': results[0], 'accuracy': results[1]}
    def get_local_label_entropy(self):
        all_labels = []; [all_labels.append(y.numpy()) for _, y in self.train_dataset]
        if not all_labels: return 0.0
        _, counts = np.unique(np.concatenate(all_labels), return_counts=True); return entropy(counts / counts.sum())


# --- MODIFIED ---
def run_caafp(num_clients=30, num_rounds=100, initial_rounds=10,
              clustering_training_rounds=30, clients_per_round=10,
              epochs_per_round=5, fine_tune_epochs=3,
              # --- NEW: Pruning schedule params ---
              pruning_frequency=5, prune_rate=0.05):
    """
    Main CA-AFP training loop with Progressive & Dynamic Pruning.
    """
    print("="*60)
    print("CA-AFP: Progressive/Dynamic Hybrid Score Pruning")
    print("="*60)
    
    # (Phase 1 & 2: Data load, Init, Clustering - unchanged)
    print("\nLoading and preparing data...");
    X_data, y_data, user_ids = load_wisdm_dataset()
    client_data = create_non_iid_data_split(X_data, y_data, user_ids, num_clients=num_clients)
    train_datasets, test_datasets = create_tf_datasets(client_data, batch_size=32)
    print("Initializing server and clients...");
    server = CAAFPServer(num_clients=num_clients, num_clusters=3)
    clients = {i: CAAFPClient(i, train_datasets[i], test_datasets[i]) for i in range(num_clients) if i in train_datasets}
    print(f"\n{'='*60}\nPHASE 1: Initial Training ({initial_rounds} rounds)\n{'='*60}")
    for round_num in range(initial_rounds):
        print(f"Round {round_num + 1}/{initial_rounds}")
        selected_clients = np.random.choice(list(clients.keys()), size=min(clients_per_round, len(clients)), replace=False)
        global_weights = server.global_model.get_weights()
        client_weights = [clients[cid].train_with_regularization(global_weights, {}, epochs=epochs_per_round) for cid in selected_clients]
        server.global_model.set_weights(server.aggregate_weights(client_weights))
    print(f"\n{'='*60}\nPHASE 2: Client Clustering\n{'='*60}")
    global_weights = server.global_model.get_weights()
    client_updates = []
    for client_id in clients.keys():
        clients[client_id].cluster_model.set_weights(global_weights)
        clients[client_id].cluster_model.fit(clients[client_id].train_dataset, epochs=1, verbose=0)
        client_updates.append(clients[client_id].compute_update(global_weights))
    server.cluster_clients(client_updates); print("Clustering complete!")
    
    # --- NEW: Set final sparsity targets *before* Phase 3 ---
    server.set_cluster_sparsity_targets(clients)
    
    print(f"\n{'='*60}")
    print("PHASE 3: Progressive Pruning & Cluster Training")
    print(f"{'='*60}")
    
    # --- MODIFIED: Phase 3 loop ---
    for round_num in range(initial_rounds, initial_rounds + clustering_training_rounds):
        print(f"\nRound {round_num + 1}/{initial_rounds + clustering_training_rounds}")
        
        # --- NEW: Check if this is a pruning round ---
        is_pruning_round = (round_num - initial_rounds) > 0 and \
                           (round_num - initial_rounds) % pruning_frequency == 0
        
        for cluster_id in range(server.num_clusters):
            cluster_client_ids = [cid for cid, clust in server.client_clusters.items() if clust == cluster_id]
            if not cluster_client_ids: continue
            
            # --- NEW: Get data needed for mask evolution ---
            if is_pruning_round:
                print(f"  Cluster {cluster_id}: Collecting data for mask evolution...")
                client_models = [clients[cid].personal_model for cid in cluster_client_ids]
                cluster_model_weights = server.cluster_models[cluster_id].get_weights()
                client_gradients = [clients[cid].compute_gradients(cluster_model_weights) for cid in cluster_client_ids]
                
                # Get current sparsity
                current_model = server.get_final_pruned_model(cluster_id)
                current_sparsity = get_model_sparsity(current_model)
                target_sparsity = server.cluster_target_sparsity[cluster_id]
                
                # --- NEW: Evolve the mask ---
                server.update_masks_and_evolve(
                    cluster_id,
                    client_models,
                    client_gradients,
                    current_sparsity,
                    target_sparsity,
                    round_num,
                    prune_rate=prune_rate
                )

            # --- Client training (now uses the mask) ---
            selected = np.random.choice(cluster_client_ids, size=min(clients_per_round // server.num_clusters + 1, len(cluster_client_ids)), replace=False)
            
            cluster_weights = server.cluster_models[cluster_id].get_weights()
            cluster_mask = server.cluster_masks[cluster_id] # Get current mask
            
            client_weights = []
            for client_id in selected:
                weights = clients[client_id].train_with_regularization(
                    cluster_weights, 
                    cluster_mask, # <-- Pass the mask
                    epochs=epochs_per_round
                )
                client_weights.append(weights)
            
            avg_weights = server.aggregate_weights(client_weights)
            server.cluster_models[cluster_id].set_weights(avg_weights)
            
            # --- NEW: Enforce mask on aggregated server model ---
            for layer in server.cluster_models[cluster_id].layers:
                if layer.name in cluster_mask:
                    layer.kernel.assign(layer.kernel.numpy() * cluster_mask[layer.name])
    
    # --- REMOVED: Phase 4 is gone ---
    
    print(f"\n{'='*60}")
    print("PHASE 5: Final Fine-Tuning & Evaluation")
    print(f"{'='*60}")
    
    # --- MODIFIED: Fine-tune on the *final* pruned models ---
    results = {}
    cluster_results = {i: [] for i in range(server.num_clusters)}
    server.final_pruned_models = {} # To store for eval

    for cluster_id in range(server.num_clusters):
        cluster_client_ids = [cid for cid, clust in server.client_clusters.items() if clust == cluster_id]
        if not cluster_client_ids: continue
        
        # Get the final model with the final mask applied
        final_pruned_model = server.get_final_pruned_model(cluster_id)
        server.final_pruned_models[cluster_id] = final_pruned_model # Save for stats
        
        print(f"  Fine-tuning clients in Cluster {cluster_id}...")
        for client_id in cluster_client_ids:
            clients[client_id].fine_tune(
                final_pruned_model.get_weights(),
                epochs=fine_tune_epochs
            )
    
    # (Evaluation loop - unchanged)
    print("\nPer-Client Results:")
    for client_id, client in clients.items():
        eval_results = client.evaluate()
        results[client_id] = eval_results
        cluster_id = server.client_clusters[client_id]
        cluster_results[cluster_id].append(eval_results['accuracy'])
        print(f"  Client {client_id:2d} (Cluster {cluster_id}): Accuracy = {eval_results['accuracy']:.4f}")
    
    # (Statistics reporting - unchanged)
    print(f"\n{'='*60}\nOverall Statistics:\n{'='*60}")
    accuracies = [r['accuracy'] for r in results.values()]
    print(f"Average Accuracy: {np.mean(accuracies):.4f}"); print(f"Std Dev: {np.std(accuracies):.4f}")
    print(f"Min: {np.min(accuracies):.4f}"); print(f"Max: {np.max(accuracies):.4f}")
    print(f"\nPer-Cluster Statistics:")
    for cluster_id in range(server.num_clusters):
        if cluster_results[cluster_id]:
            if cluster_id in server.final_pruned_models:
                cluster_sparsity = get_model_sparsity(server.final_pruned_models[cluster_id])
            else: cluster_sparsity = 0.0
            print(f"  Cluster {cluster_id}:"); print(f"    Clients: {len(cluster_results[cluster_id])}")
            print(f"    Avg Acc: {np.mean(cluster_results[cluster_id]):.4f}"); print(f"    Std Dev: {np.std(cluster_results[cluster_id]):.4f}")
            print(f"    Sparsity: {cluster_sparsity:.2%}")
    
    return server, clients, results

# --- (Main execution block - unchanged from final_2.py) ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    SEED = args.seed
    import random
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED); os.environ['TF_DETERMINISTIC_OPS'] = '1'

    run_id = f"caafp_hybrid_70_seed_{SEED}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs('logs', exist_ok=True); os.makedirs('saved_models', exist_ok=True)
    log_filepath = f'logs/run_log_{run_id}.log'
    
    original_stdout = sys.stdout
    logger = Logger(log_filepath)
    sys.stdout = logger
    print(f"--- Starting CA-AFP Hybrid Run: {run_id} with seed {SEED} ---")
    
    try:
        server, clients, results = run_caafp(
            num_clients=30, num_rounds=100, initial_rounds=10,
            clustering_training_rounds=30, clients_per_round=10,
            epochs_per_round=3, fine_tune_epochs=3,
            pruning_frequency=5, prune_rate=0.05 # <-- New params
        )
        print(f"\n--- Saving models for run {run_id} ---")
        for cluster_id, model in server.final_pruned_models.items():
            save_path = f'saved_models/model_{run_id}_cluster_{cluster_id}.h5'
            model.save_weights(save_path); print(f"Saved model for cluster {cluster_id} to {save_path}")
    finally:
        sys.stdout = original_stdout; del logger
        print(f"Run {run_id} complete. Logs saved to {log_filepath}")
