"""
FLCAP Baseline (MASK CLUSTERING VERSION)
========================================
This is an ablation study version of FLCAP.

It implements the clustering method described in the FLCAP paper:
- Clients compute local masks based on magnitude [cite: 177]
- Server clusters clients based on the HAMMING DISTANCE of these masks 

All other parameters (model, data, training) are
identical to the main benchmark for a fair comparison.
"""
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ON_NN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from sklearn.cluster import AgglomerativeClustering
# NEW: Import Hamming distance
from scipy.spatial.distance import pdist, squareform
import sys
from datetime import datetime
from data_loader_uci import load_uci_har_dataset, create_non_iid_data_split, create_tf_datasets
from models import create_lstm_model, get_model_sparsity

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

class FLCAPServer:
    def __init__(self, num_clients, num_clusters=3, base_sparsity=0.7):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.base_sparsity = base_sparsity
        self.cluster_sparsity = {c: base_sparsity for c in range(num_clusters)}
        self.global_model = create_lstm_model(input_shape=(128, 9))
        self.cluster_models = {}
        self.pruned_models = {}
        self.client_clusters = {}
        self.clustered = False

    def aggregate_weights(self, client_weights):
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        for weights in client_weights:
            for i, w in enumerate(weights): avg_weights[i] += w
        for i in range(len(avg_weights)): avg_weights[i] /= len(client_weights)
        return avg_weights

    # --- NEW: CLUSTERING BY MASK SIMILARITY (HAMMING DISTANCE) ---
    def cluster_clients_by_masks(self, client_masks):
        """
        Cluster clients based on mask similarity using Hamming distance,
        as described in the FLCAP paper (Sec III-C).
        """
        # client_masks is a list of flat binary numpy arrays
        client_masks_array = np.array(client_masks)
        
        # Calculate pairwise Hamming distance
        # pdist returns a condensed 1D array
        distance_vector = pdist(client_masks_array, 'hamming')
        
        # Convert to a square matrix for the clustering algorithm
        distance_matrix = squareform(distance_vector)
        
        # Cluster using the precomputed distance matrix
        try:
            clustering = AgglomerativeClustering(
                n_clusters=self.num_clusters,
                metric='precomputed',
                linkage='complete' # We use 'average' as in our final tests
            )
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=self.num_clusters,
                affinity='precomputed',
                linkage='complete'
            )
        
        cluster_labels = clustering.fit_predict(distance_matrix)

        # Store assignments
        for client_id, cluster_id in enumerate(cluster_labels):
            self.client_clusters[client_id] = int(cluster_id)

        # Initialize dense cluster models
        for cluster_id in range(self.num_clusters):
            self.cluster_models[cluster_id] = create_lstm_model(input_shape=(128, 9))
            self.cluster_models[cluster_id].set_weights(self.global_model.get_weights())

        self.clustered = True
        return cluster_labels
    # --- END NEW FUNCTION ---

    def set_adaptive_sparsity(self, clients, s_min=0.5, s_max=0.9):
        print("\n--- FIXED 70% SPARSITY TEST ---")
        for c in range(self.num_clusters):
            self.cluster_sparsity[c] = 0.7
            print(f"  Cluster {c}: sparsity=70.00%")

    def apply_magnitude_pruning_progressive(self, cluster_id, steps=3, fine_tune_epochs=1, clients=None):
        print(f"\n--- Applying Global Magnitude Pruning to Cluster {cluster_id} ---")
        print(f"  Target sparsity: {self.cluster_sparsity[cluster_id]:.2%}")
        print(f"  Progressive steps: {steps}")
        
        dense_model = self.cluster_models[cluster_id]
        pruned_model = create_lstm_model(input_shape=(128, 9))
        pruned_model.set_weights(dense_model.get_weights())

        flat = []
        refs = []
        for layer in pruned_model.layers:
            if hasattr(layer, 'kernel'):
                w = layer.kernel.numpy()
                refs.append((layer, w.shape))
                flat.append(w.flatten())

        if not flat:
            self.pruned_models[cluster_id] = pruned_model
            return pruned_model

        flat = np.concatenate(flat)
        scores = np.abs(flat).astype(np.float32)
        target = self.cluster_sparsity[cluster_id]

        for t in range(1, steps + 1):
            frac = target * (t / steps)
            k = int(len(scores) * frac)
            thresh = np.partition(scores, k)[k] if k > 0 and k < len(scores) else -np.inf
            mask_all = (scores > thresh).astype(flat.dtype)

            cursor = 0
            for layer, shape in refs:
                size = int(np.prod(shape))
                seg_mask = mask_all[cursor:cursor + size]
                seg_vals = flat[cursor:cursor + size]
                masked = seg_vals * seg_mask
                layer.kernel.assign(masked.reshape(shape))
                cursor += size

            print(f"  Step {t}/{steps}: pruned to {frac:.2%}")

            if clients is not None and fine_tune_epochs > 0 and t < steps:
                cluster_cids = [cid for cid, cl in self.client_clusters.items() if cl == cluster_id]
                for cid in cluster_cids:
                    clients[cid].fine_tune(pruned_model.get_weights(), epochs=fine_tune_epochs)

        self.pruned_models[cluster_id] = pruned_model
        actual_sparsity = get_model_sparsity(pruned_model)
        print(f"  Final sparsity: {actual_sparsity:.2%}")
        return pruned_model


class FLCAPClient:
    def __init__(self, client_id, train_dataset, test_dataset, lambda_reg=0.01):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lambda_reg = lambda_reg
        self.personal_model = create_lstm_model(input_shape=(128, 9))
        self.cluster_model = create_lstm_model(input_shape=(128, 9))

    # --- NEW: FUNCTION TO COMPUTE LOCAL MASK FOR CLUSTERING ---
    def compute_local_mask(self, global_weights, epochs=1, sparsity=0.7):
        """
        Train locally for e1 epochs and compute a mask[cite: 176, 177].
        This implements Algorithm 1, steps 4-5.
        """
        # 1. Set model to global weights
        self.cluster_model.set_weights(global_weights)
        
        # 2. Train locally for e1 epochs (ModelUpdate)
        self.cluster_model.fit(self.train_dataset, epochs=epochs, verbose=0)
        
        # 3. Compute mask based on local model's magnitudes (ComputeMask)
        flat = []
        for layer in self.cluster_model.layers:
            if hasattr(layer, 'kernel'):
                w = layer.kernel.numpy()
                flat.append(w.flatten())
        
        if not flat:
            return np.array([])
            
        flat = np.concatenate(flat)
        scores = np.abs(flat).astype(np.float32)
        
        # 4. Find threshold for target sparsity
        k = int(len(scores) * sparsity)
        thresh = np.partition(scores, k)[k] if k > 0 and k < len(scores) else -np.inf
        
        # 5. Return flat binary mask
        mask_all = (scores > thresh).astype(np.int8)
        return mask_all
    # --- END NEW FUNCTION ---

    def train_with_regularization(self, cluster_weights, epochs=5):
        self.cluster_model.set_weights(cluster_weights)
        for epoch in range(epochs):
            for x, y in self.train_dataset:
                with tf.GradientTape() as tape:
                    predictions = self.personal_model(x, training=True)
                    local_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, predictions))
                    reg_loss = 0.0
                    for pw, cw in zip(self.personal_model.trainable_weights, self.cluster_model.trainable_weights):
                        reg_loss += tf.reduce_sum(tf.square(pw - cw))
                    reg_loss = (self.lambda_reg / 2.0) * reg_loss
                    total_loss = local_loss + reg_loss
                gradients = tape.gradient(total_loss, self.personal_model.trainable_weights)
                self.personal_model.optimizer.apply_gradients(zip(gradients, self.personal_model.trainable_weights))
            self.cluster_model.fit(self.train_dataset, epochs=1, verbose=0)
        return self.cluster_model.get_weights()

    def fine_tune(self, pruned_model_weights, epochs=3):
        self.personal_model.set_weights(pruned_model_weights)
        self.personal_model.fit(self.train_dataset, epochs=epochs, verbose=0)

    # Note: compute_update is no longer needed for clustering
    def evaluate(self):
        results = self.personal_model.evaluate(self.test_dataset, verbose=0)
        return {'loss': results[0], 'accuracy': results[1]}


def run_flcap(num_clients=30, num_rounds=100, initial_rounds=10,
              clustering_training_rounds=30, clients_per_round=10,
              epochs_per_round=5, fine_tune_epochs=3,
              base_sparsity=0.7, random_seed=42):
    
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    print("="*60)
    print("FLCAP: ABLATION (Mask-Based Clustering)")
    print("="*60)
    print(f"Random seed: {random_seed}")

    print("\nLoading and preparing data...")
    X_data, y_data, user_ids = load_uci_har_dataset()
    client_data = create_non_iid_data_split(X_data, y_data, user_ids, num_clients=num_clients)
    train_datasets, test_datasets = create_tf_datasets(client_data, batch_size=32)

    print("Initializing server and clients...")
    server = FLCAPServer(num_clients=num_clients, num_clusters=3, base_sparsity=base_sparsity)
    clients = {i: FLCAPClient(i, train_datasets[i], test_datasets[i]) for i in range(num_clients) if i in train_datasets}

    print(f"\n{'='*60}\nPHASE 1: Initial Global Training\n{'='*60}")
    for round_num in range(initial_rounds):
        print(f"Round {round_num + 1}/{initial_rounds}")
        selected_clients = np.random.choice(list(clients.keys()), size=min(clients_per_round, len(clients)), replace=False)
        global_weights = server.global_model.get_weights()
        client_weights = [clients[cid].train_with_regularization(global_weights, epochs=epochs_per_round) for cid in selected_clients]
        server.global_model.set_weights(server.aggregate_weights(client_weights))

    print(f"\n{'='*60}\nPHASE 2: Client Clustering (Mask-Based)\n{'='*60}")
    
    # --- MODIFIED: PHASE 2 ---
    print(f"\nPerforming client clustering (based on local masks)...")
    global_weights = server.global_model.get_weights()
    client_masks = []
    
    # Each client computes its local mask [cite: 177]
    for client_id in clients.keys():
        mask = clients[client_id].compute_local_mask(
            global_weights, 
            epochs=1, # e1 epochs
            sparsity=base_sparsity
        )
        client_masks.append(mask)

    # Server clusters clients based on mask similarity [cite: 178]
    cluster_labels = server.cluster_clients_by_masks(client_masks)
    # --- END MODIFIED PHASE 2 ---
    
    print(f"Clustering complete!")
    for cluster_id in range(server.num_clusters):
        cluster_clients = [cid for cid, clust in server.client_clusters.items() if clust == cluster_id]
        print(f"  Cluster {cluster_id}: {len(cluster_clients)} clients")

    print(f"\n{'='*60}\nPHASE 3: Training Dense Cluster Models\n{'='*60}")
    for round_num in range(initial_rounds, initial_rounds + clustering_training_rounds):
        print(f"\nRound {round_num + 1}/{initial_rounds + clustering_training_rounds}")
        for cluster_id in range(server.num_clusters):
            cluster_client_ids = [cid for cid, clust in server.client_clusters.items() if clust == cluster_id]
            if not cluster_client_ids: continue
            selected = np.random.choice(cluster_client_ids, size=min(clients_per_round // server.num_clusters + 1, len(cluster_client_ids)), replace=False)
            cluster_weights = server.cluster_models[cluster_id].get_weights()
            client_weights = [clients[cid].train_with_regularization(cluster_weights, epochs=epochs_per_round) for cid in selected]
            server.cluster_models[cluster_id].set_weights(server.aggregate_weights(client_weights))

    print(f"\n{'='*60}\nPHASE 4: Adaptive Progressive Pruning\n{'='*60}")
    server.set_adaptive_sparsity(clients)
    for cluster_id in range(server.num_clusters):
        cluster_client_ids = [cid for cid, clust in server.client_clusters.items() if clust == cluster_id]
        if not cluster_client_ids: continue
        pruned_model = server.apply_magnitude_pruning_progressive(
            cluster_id, steps=3, fine_tune_epochs=1, clients=clients
        )
        print(f"  Final fine-tuning for cluster {cluster_id}...")
        for client_id in cluster_client_ids:
            clients[client_id].fine_tune(pruned_model.get_weights(), epochs=fine_tune_epochs)

    print(f"\n{'='*60}\nPHASE 5: Final Evaluation\n{'='*60}")
    results = {}
    cluster_results = {i: [] for i in range(server.num_clusters)}
    for client_id, client in clients.items():
        eval_results = client.evaluate()
        results[client_id] = eval_results
        cluster_id = server.client_clusters[client_id]
        cluster_results[cluster_id].append(eval_results['accuracy'])

    print("\nPer-Client Results:")
    for client_id, result in results.items():
        cluster_id = server.client_clusters[client_id]
        print(f"  Client {client_id:2d} (Cluster {cluster_id}): Accuracy = {result['accuracy']:.4f}")

    print(f"\n{'='*60}\nOverall Statistics:\n{'='*60}")
    accuracies = [r['accuracy'] for r in results.values()]
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Dev: {np.std(accuracies):.4f}")
    print(f"Min: {np.min(accuracies):.4f}")
    print(f"Max: {np.max(accuracies):.4f}")

    print(f"\nPer-Cluster Statistics:")
    for cluster_id in range(server.num_clusters):
        if cluster_results[cluster_id]:
            cluster_acc = cluster_results[cluster_id]
            cluster_sparsity = get_model_sparsity(server.pruned_models.get(cluster_id, create_lstm_model(input_shape=(128, 9))))
            print(f"  Cluster {cluster_id}:")
            print(f"    Clients: {len(cluster_acc)}")
            print(f"    Avg Acc: {np.mean(cluster_acc):.4f}")
            print(f"    Std Dev: {np.std(cluster_acc):.4f}")
            print(f"    Actual Sparsity: {cluster_sparsity:.2%}")

    return server, clients, results

if __name__ == "__main__":
    import argparse
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    SEED = args.seed
    
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED); os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    run_id = f"flcap_MASK_CLUSTER_seed_{SEED}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs('logs_tuning', exist_ok=True)
    log_filepath = f'logs_tuning/run_log_{run_id}.log'
    
    original_stdout = sys.stdout
    logger = Logger(log_filepath)
    sys.stdout = logger

    print(f"--- Starting FLCAP (Mask Clustering) Run: {run_id} with seed {SEED} ---")
    try:
        server, clients, results = run_flcap(
            num_clients=30,
            num_rounds=100,
            initial_rounds=10,
            clustering_training_rounds=30,
            clients_per_round=10,
            epochs_per_round=3,
            fine_tune_epochs=3,
            base_sparsity=0.7,
            random_seed=SEED
        )
    finally:
        sys.stdout = original_stdout
        del logger
        print(f"Run {run_id} complete. Logs saved to {log_filepath}")
