"""
FLCAP Baseline Implementation (UPDATED)
========================================
Paper: "FLCAP: Federated Learning with Clustered Adaptive Pruning" (2023)
Authors: Miralles et al.

Updates in this version:
- ADAPTIVE per-cluster sparsity based on validation loss
- GLOBAL magnitude threshold across all weights (not per-layer)
- PROGRESSIVE pruning in K steps with inter-step fine-tuning
- Fixed training loop (epochs=1 in fit) for correct compute budget
- sklearn clustering compatibility (metric vs affinity)

Fair Comparison Setup:
- Same model architecture (LSTM-based HAR)
- Same dataset splits (WISDM with your non-IID distribution)
- Same FL parameters (30 clients, 10 clients/round, 3 local epochs)
- Same total training budget (40 rounds dense + 3 epochs FT)
- Same clustering algorithm (Agglomerative on cosine similarity)
"""
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Use GPU 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN optimization
import numpy as np
import tensorflow as tf
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
import copy
import sys
import os
from datetime import datetime
from data_loader import load_wisdm_dataset, create_non_iid_data_split, create_tf_datasets
from models import create_lstm_model

class Logger:
    """Redirects print statements to both terminal and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
import numpy as np
import tensorflow as tf
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
import sys
import os
from datetime import datetime
from data_loader import load_wisdm_dataset, create_non_iid_data_split, create_tf_datasets
from models import create_lstm_model, get_model_sparsity

class Logger:
    """Redirects print statements to both terminal and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def __del__(self):
        # Restore stdout and close file when object is destroyed
        sys.stdout = self.terminal
        if self.logfile:
            self.logfile.close()

class FLCAPServer:
    """Server for FLCAP framework with adaptive sparsity."""
    def __init__(self, num_clients, num_clusters=3, base_sparsity=0.7):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.base_sparsity = base_sparsity
        self.cluster_sparsity = {c: base_sparsity for c in range(num_clusters)}  # Adaptive per cluster
        self.global_model = create_lstm_model()
        self.cluster_models = {}  # Dense cluster models
        self.pruned_models = {}   # Pruned cluster models
        self.client_clusters = {}
        self.clustered = False

    def aggregate_weights(self, client_weights):
        """FedAvg aggregation."""
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        for weights in client_weights:
            for i, w in enumerate(weights):
                avg_weights[i] += w
        for i in range(len(avg_weights)):
            avg_weights[i] /= len(client_weights)
        return avg_weights

    def cluster_clients(self, client_updates):
        """Cluster clients based on model update similarity (same as CA-AFP)."""
        # Flatten updates
        flattened_updates = []
        for update in client_updates:
            flattened = np.concatenate([u.flatten() for u in update])
            flattened_updates.append(flattened)
        flattened_updates = np.array(flattened_updates)

        # Calculate cosine similarity
        n_clients = len(flattened_updates)
        similarity_matrix = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(n_clients):
                if i != j:
                    similarity = 1 - cosine(flattened_updates[i], flattened_updates[j])
                    similarity_matrix[i, j] = similarity
                else:
                    similarity_matrix[i, j] = 1.0

        # Cluster
        distance_matrix = 1 - similarity_matrix
        
        # FIXED: sklearn compatibility (metric vs affinity)
        try:
            clustering = AgglomerativeClustering(
                n_clusters=self.num_clusters,
                metric='precomputed',
                linkage='average'
            )
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=self.num_clusters,
                affinity='precomputed',
                linkage='average'
            )
        
        cluster_labels = clustering.fit_predict(distance_matrix)

        # Store assignments
        for client_id, cluster_id in enumerate(cluster_labels):
            self.client_clusters[client_id] = int(cluster_id)

        # Initialize dense cluster models
        for cluster_id in range(self.num_clusters):
            self.cluster_models[cluster_id] = create_lstm_model()
            self.cluster_models[cluster_id].set_weights(self.global_model.get_weights())

        self.clustered = True
        return cluster_labels

    def set_adaptive_sparsity(self, clients, s_min=0.5, s_max=0.9):
        """
        MODIFIED: Forcing fixed 70% sparsity for all clusters.
        """
        print("\n--- FIXED 70% SPARSITY TEST ---")
        for c in range(self.num_clusters):
            self.cluster_sparsity[c] = 0.7
            print(f"  Cluster {c}: sparsity=70.00%")
    def apply_magnitude_pruning_progressive(self, cluster_id, steps=3, fine_tune_epochs=1, clients=None):
        """
        NEW: Apply GLOBAL magnitude threshold pruning progressively in K steps.
        Inter-step fine-tuning stabilizes accuracy.
        """
        print(f"\n--- Applying Global Magnitude Pruning to Cluster {cluster_id} ---")
        print(f"  Target sparsity: {self.cluster_sparsity[cluster_id]:.2%}")
        print(f"  Progressive steps: {steps}")
        
        dense_model = self.cluster_models[cluster_id]
        pruned_model = create_lstm_model()
        pruned_model.set_weights(dense_model.get_weights())

        # Collect all prunable weights across the model
        flat = []
        refs = []  # (layer, shape) references
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

        # Progressive pruning in K steps
        for t in range(1, steps + 1):
            frac = target * (t / steps)
            k = int(len(scores) * frac)
            thresh = np.partition(scores, k)[k] if k > 0 and k < len(scores) else -np.inf
            mask_all = (scores > thresh).astype(flat.dtype)

            # Write masked weights back to each layer
            cursor = 0
            for layer, shape in refs:
                size = int(np.prod(shape))
                seg_mask = mask_all[cursor:cursor + size]
                seg_vals = flat[cursor:cursor + size]
                masked = seg_vals * seg_mask
                layer.kernel.assign(masked.reshape(shape))
                cursor += size

            print(f"  Step {t}/{steps}: pruned to {frac:.2%}")

            # Inter-step fine-tuning with cluster clients
            if clients is not None and fine_tune_epochs > 0 and t < steps:
                cluster_cids = [cid for cid, cl in self.client_clusters.items() if cl == cluster_id]
                for cid in cluster_cids:
                    clients[cid].fine_tune(pruned_model.get_weights(), epochs=fine_tune_epochs)

        self.pruned_models[cluster_id] = pruned_model
        actual_sparsity = get_model_sparsity(pruned_model)
        print(f"  Final sparsity: {actual_sparsity:.2%}")
        return pruned_model


class FLCAPClient:
    """Client for FLCAP framework."""
    def __init__(self, client_id, train_dataset, test_dataset, lambda_reg=0.01):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lambda_reg = lambda_reg
        self.personal_model = create_lstm_model()
        self.cluster_model = create_lstm_model()

    def train_with_regularization(self, cluster_weights, epochs=5):
        """
        Train with regularization to cluster model (same as CA-AFP).
        FIXED: epochs=1 in fit to avoid double-counting.
        """
        self.cluster_model.set_weights(cluster_weights)
        
        for epoch in range(epochs):
            for x, y in self.train_dataset:
                with tf.GradientTape() as tape:
                    predictions = self.personal_model(x, training=True)
                    local_loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
                    local_loss = tf.reduce_mean(local_loss)

                    # Regularization term
                    reg_loss = 0.0
                    for pw, cw in zip(self.personal_model.trainable_weights,
                                     self.cluster_model.trainable_weights):
                        diff = pw - cw
                        reg_loss += tf.reduce_sum(tf.square(diff))
                    reg_loss = (self.lambda_reg / 2.0) * reg_loss

                    total_loss = local_loss + reg_loss

                gradients = tape.gradient(total_loss, self.personal_model.trainable_weights)
                self.personal_model.optimizer.apply_gradients(
                    zip(gradients, self.personal_model.trainable_weights)
                )

            # FIXED: epochs=1 in fit (was epochs=epochs causing double-counting)
            self.cluster_model.fit(self.train_dataset, epochs=1, verbose=0)

        return self.cluster_model.get_weights()

    def fine_tune(self, pruned_model_weights, epochs=3):
        """Fine-tune the pruned model."""
        self.personal_model.set_weights(pruned_model_weights)
        self.personal_model.fit(self.train_dataset, epochs=epochs, verbose=0)

    def compute_update(self, old_weights):
        """Compute model update."""
        new_weights = self.cluster_model.get_weights()
        update = [new - old for new, old in zip(new_weights, old_weights)]
        return update

    def evaluate(self):
        """Evaluate personal model."""
        results = self.personal_model.evaluate(self.test_dataset, verbose=0)
        return {'loss': results[0], 'accuracy': results[1]}


def run_flcap(num_clients=30, num_rounds=100, initial_rounds=10,
              clustering_training_rounds=30, clients_per_round=10,
              epochs_per_round=5, fine_tune_epochs=3,
              base_sparsity=0.7, random_seed=42):
    """
    Main FLCAP training loop with adaptive sparsity and progressive pruning.
    
    Args:
        base_sparsity: Base sparsity (will be adapted per cluster)
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    print("="*60)
    print("FLCAP: Federated Learning with Clustered Adaptive Pruning")
    print("="*60)
    print(f"Random seed: {random_seed}")
    print(f"Base sparsity: {base_sparsity:.1%} (will be adapted per cluster)")

    print("\nLoading and preparing data...")
    X_data, y_data, user_ids = load_wisdm_dataset()
    client_data = create_non_iid_data_split(X_data, y_data, user_ids, num_clients=num_clients)
    train_datasets, test_datasets = create_tf_datasets(client_data, batch_size=32)

    # Initialize server and clients
    print("Initializing server and clients...")
    server = FLCAPServer(num_clients=num_clients, num_clusters=3, base_sparsity=base_sparsity)

    clients = {
        i: FLCAPClient(i, train_datasets[i], test_datasets[i], lambda_reg=0.01)
        for i in range(num_clients) if i in train_datasets
    }

    print(f"\n{'='*60}")
    print("PHASE 1: Initial Global Training")
    print(f"{'='*60}")

    # Phase 1: Initial FL rounds (same as CA-AFP)
    print(f"\nRunning {initial_rounds} initial FL rounds...")
    for round_num in range(initial_rounds):
        print(f"Round {round_num + 1}/{initial_rounds}")

        selected_clients = np.random.choice(
            list(clients.keys()),
            size=min(clients_per_round, len(clients)),
            replace=False
        )

        global_weights = server.global_model.get_weights()
        client_weights = []

        for client_id in selected_clients:
            weights = clients[client_id].train_with_regularization(
                global_weights, epochs=epochs_per_round
            )
            client_weights.append(weights)

        avg_weights = server.aggregate_weights(client_weights)
        server.global_model.set_weights(avg_weights)

    print(f"\n{'='*60}")
    print("PHASE 2: Client Clustering")
    print(f"{'='*60}")

    # Phase 2: Clustering (same as CA-AFP)
    print(f"\nPerforming client clustering...")
    global_weights = server.global_model.get_weights()
    client_updates = []

    for client_id in clients.keys():
        clients[client_id].cluster_model.set_weights(global_weights)
        clients[client_id].cluster_model.fit(
            clients[client_id].train_dataset, epochs=1, verbose=0
        )
        update = clients[client_id].compute_update(global_weights)
        client_updates.append(update)

    cluster_labels = server.cluster_clients(client_updates)
    print(f"Clustering complete!")
    for cluster_id in range(server.num_clusters):
        cluster_clients = [cid for cid, clust in server.client_clusters.items()
                          if clust == cluster_id]
        print(f"  Cluster {cluster_id}: {len(cluster_clients)} clients")

    print(f"\n{'='*60}")
    print("PHASE 3: Training Dense Cluster Models")
    print(f"{'='*60}")

    # Phase 3: Train specialized dense models for each cluster (same as CA-AFP)
    for round_num in range(initial_rounds, initial_rounds + clustering_training_rounds):
        print(f"\nRound {round_num + 1}/{initial_rounds + clustering_training_rounds}")

        for cluster_id in range(server.num_clusters):
            cluster_client_ids = [
                cid for cid, clust in server.client_clusters.items()
                if clust == cluster_id
            ]

            if not cluster_client_ids:
                continue

            selected = np.random.choice(
                cluster_client_ids,
                size=min(clients_per_round // server.num_clusters + 1,
                        len(cluster_client_ids)),
                replace=False
            )

            cluster_weights = server.cluster_models[cluster_id].get_weights()
            client_weights = []

            for client_id in selected:
                weights = clients[client_id].train_with_regularization(
                    cluster_weights, epochs=epochs_per_round
                )
                client_weights.append(weights)

            # Update cluster model
            avg_weights = server.aggregate_weights(client_weights)
            server.cluster_models[cluster_id].set_weights(avg_weights)

    print(f"\n{'='*60}")
    print("PHASE 4: Adaptive Progressive Pruning")
    print(f"{'='*60}")

    # NEW: Set adaptive sparsity per cluster based on validation loss
    server.set_adaptive_sparsity(clients, s_min=0.5, s_max=0.9)

    # Phase 4: Apply global-threshold progressive pruning to each cluster
    for cluster_id in range(server.num_clusters):
        cluster_client_ids = [
            cid for cid, clust in server.client_clusters.items()
            if clust == cluster_id
        ]

        if not cluster_client_ids:
            continue

        # NEW: Progressive global-threshold magnitude pruning
        pruned_model = server.apply_magnitude_pruning_progressive(
            cluster_id, steps=3, fine_tune_epochs=1, clients=clients
        )

        # Fine-tune the pruned model with cluster clients
        print(f"  Final fine-tuning for cluster {cluster_id}...")
        for client_id in cluster_client_ids:
            clients[client_id].fine_tune(
                pruned_model.get_weights(),
                epochs=fine_tune_epochs
            )

    print(f"\n{'='*60}")
    print("PHASE 5: Final Evaluation")
    print(f"{'='*60}")

    # Evaluate all clients
    results = {}
    cluster_results = {i: [] for i in range(server.num_clusters)}

    for client_id, client in clients.items():
        eval_results = client.evaluate()
        results[client_id] = eval_results
        cluster_id = server.client_clusters[client_id]
        cluster_results[cluster_id].append(eval_results['accuracy'])

    # Print detailed results
    print("\nPer-Client Results:")
    for client_id, result in results.items():
        cluster_id = server.client_clusters[client_id]
        print(f"  Client {client_id:2d} (Cluster {cluster_id}): "
              f"Accuracy = {result['accuracy']:.4f}")

    # Overall statistics
    print(f"\n{'='*60}")
    print("Overall Statistics:")
    print(f"{'='*60}")
    accuracies = [r['accuracy'] for r in results.values()]
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Dev: {np.std(accuracies):.4f}")
    print(f"Min: {np.min(accuracies):.4f}")
    print(f"Max: {np.max(accuracies):.4f}")

    # Per-cluster statistics
    print(f"\nPer-Cluster Statistics:")
    for cluster_id in range(server.num_clusters):
        if cluster_results[cluster_id]:
            cluster_acc = cluster_results[cluster_id]
            if cluster_id in server.pruned_models:
                cluster_sparsity = get_model_sparsity(
                    server.pruned_models[cluster_id]
                )
            else:
                cluster_sparsity = 0.0

            print(f"  Cluster {cluster_id}:")
            print(f"    Clients: {len(cluster_acc)}")
            print(f"    Avg Acc: {np.mean(cluster_acc):.4f}")
            print(f"    Std Dev: {np.std(cluster_acc):.4f}")
            print(f"    Adaptive Sparsity: {server.cluster_sparsity[cluster_id]:.2%}")
            print(f"    Actual Sparsity: {cluster_sparsity:.2%}")

    return server, clients, results


if __name__ == "__main__":
    import sys
    from datetime import datetime
    import argparse  # <-- ADD THIS
    import random    # <-- ADD THIS
    import os        # <-- ADD THIS
    run_id_base = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('logs', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    SEED = args.seed
    
    # Set all seeds
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Run with same parameters as CA-AFP for fair comparison
    # Test multiple base sparsity levels
    for i, sparsity in enumerate([0.7]):
        # --- Setup logging and paths FOR THIS LOOP ---
        run_id = f"flcap_70_seed_{SEED}_sparsity_{int(sparsity*100)}"
        log_filepath = f'logs/run_log_{run_id}.log'
        model_save_path = f'saved_models/model_{run_id}'

        original_stdout = sys.stdout
        logger = Logger(log_filepath)
        sys.stdout = logger

        print(f"--- Starting FLCAP Run: {run_id} with seed {SEED} ---")
        try:
            # --- Run the experiment ---
            server, clients, results = run_flcap(
                num_clients=30,
                num_rounds=100,
                initial_rounds=10,
                clustering_training_rounds=30,
                clients_per_round=10,
                epochs_per_round=3,
                fine_tune_epochs=3,
                base_sparsity=sparsity,
                random_seed=SEED  # <-- USE THE SEED
            )

            # --- Save the models ---
            print(f"\n--- Saving models for run {run_id} ---")
            for cluster_id, model in server.pruned_models.items():
                save_path = f"{model_save_path}_cluster_{cluster_id}.h5"
                model.save_weights(save_path)
                print(f"Saved model for cluster {cluster_id} to {save_path}")

        finally:
            # --- Restore stdout and close log ---
            sys.stdout = original_stdout
            del logger
            print(f"Run {run_id} complete. Logs saved to {log_filepath}")
