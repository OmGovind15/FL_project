import numpy as np
import tensorflow as tf
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from scipy.stats import entropy

from data_loader import load_wisdm_dataset, create_non_iid_data_split, create_tf_datasets
from models import create_lstm_model, create_prunable_lstm_model, get_model_sparsity

class ClusterAwareImportanceScorer:
    """
    Novelty 1: Calculate hybrid importance scores for weights.
    Score = alpha * Magnitude + beta * Coherence + gamma * Consistency
    """
    def __init__(self, alpha=0.5, beta=0.25, gamma=0.25):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def calculate_scores(self, dense_cluster_model, client_models_in_cluster):
        """
        Calculate hybrid importance scores for all weights.
        
        Returns:
            dict: layer_name -> importance scores (same shape as weights)
        """
        importance_scores = {}
        
        for layer in dense_cluster_model.layers:
            if not hasattr(layer, 'kernel'):
                continue
            
            layer_name = layer.name
            cluster_weights = layer.kernel.numpy()
            
            # 1. Magnitude scores (normalized)
            magnitude_scores = np.abs(cluster_weights)
            mag_max = np.max(magnitude_scores)
            if mag_max > 0:
                magnitude_scores = magnitude_scores / mag_max
            
            # 2. Cluster coherence scores (low variance = high coherence)
            client_weights_list = []
            for client_model in client_models_in_cluster:
                for client_layer in client_model.layers:
                    if client_layer.name == layer_name and hasattr(client_layer, 'kernel'):
                        client_weights_list.append(client_layer.kernel.numpy())
                        break
            
            if client_weights_list:
                client_weights_array = np.array(client_weights_list)
                # Calculate variance across clients (axis=0)
                variance = np.var(client_weights_array, axis=0)
                # Coherence is inverse of variance (normalized)
                coherence_scores = 1.0 / (1.0 + variance)
                coh_max = np.max(coherence_scores)
                if coh_max > 0:
                    coherence_scores = coherence_scores / coh_max
            else:
                coherence_scores = np.ones_like(cluster_weights)
            
            # 3. Gradient consistency scores
            # For simplicity, we'll use a placeholder here
            # In practice, you'd collect gradients from clients
            consistency_scores = np.ones_like(cluster_weights)
            
            # Combine scores
            hybrid_score = (self.alpha * magnitude_scores + 
                           self.beta * coherence_scores + 
                           self.gamma * consistency_scores)
            
            importance_scores[layer_name] = hybrid_score
        
        return importance_scores


class AdaptivePruningScheduler:
    """
    Novelty 2: Calculate adaptive sparsity targets for each cluster.
    """
    def __init__(self, base_sparsity=0.7, max_sparsity=0.9, min_sparsity=0.5):
        self.base_sparsity = base_sparsity
        self.max_sparsity = max_sparsity
        self.min_sparsity = min_sparsity
    
    def get_sparsity_for_cluster(self, cluster_data_labels, num_clients_in_cluster):
        """
        Calculate adaptive sparsity based on data complexity and cluster size.
        
        Args:
            cluster_data_labels: array of activity labels in cluster
            num_clients_in_cluster: number of clients in cluster
        
        Returns:
            float: target sparsity (0-1)
        """
        # 1. Calculate label entropy (data complexity)
        unique_labels, counts = np.unique(cluster_data_labels, return_counts=True)
        probabilities = counts / counts.sum()
        label_entropy = entropy(probabilities)
        
        # Normalize entropy (max entropy for 6 classes is log(6))
        max_entropy = np.log(6)
        normalized_entropy = label_entropy / max_entropy if max_entropy > 0 else 0
        
        # 2. Factor in cluster size
        # Larger clusters get lower sparsity (less aggressive pruning)
        size_factor = min(num_clients_in_cluster / 10.0, 1.0)
        
        # 3. Calculate target sparsity
        # High entropy (complex) -> lower sparsity
        # Large cluster -> lower sparsity
        complexity_adjustment = normalized_entropy * 0.2  # Up to 20% reduction
        size_adjustment = size_factor * 0.1  # Up to 10% reduction
        
        target_sparsity = self.base_sparsity - complexity_adjustment - size_adjustment
        target_sparsity = np.clip(target_sparsity, self.min_sparsity, self.max_sparsity)
        
        return target_sparsity


class CAAFPServer:
    """Server for CA-AFP framework."""
    def __init__(self, num_clients, num_clusters=3):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.global_model = create_lstm_model()
        self.cluster_models = {}  # Dense cluster models
        self.final_pruned_models = {}  # Final pruned models
        self.client_clusters = {}
        self.clustered = False
        
        # Novel components
        self.importance_scorer = ClusterAwareImportanceScorer(
            alpha=0.5, beta=0.25, gamma=0.25
        )
        self.pruning_scheduler = AdaptivePruningScheduler(
            base_sparsity=0.7, max_sparsity=0.9, min_sparsity=0.5
        )
    
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
        """Cluster clients based on model update similarity."""
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
        clustering = AgglomerativeClustering(
            n_clusters=self.num_clusters,
            metric='precomputed',
            linkage='complete'
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
    
    def apply_cluster_aware_pruning(self, cluster_id, client_models_in_cluster, 
                                    cluster_data_labels):
        """
        Apply novel cluster-aware adaptive pruning to a cluster.
        """
        print(f"\n  Applying CA-AFP to Cluster {cluster_id}...")
        
        dense_model = self.cluster_models[cluster_id]
        num_clients = len(client_models_in_cluster)
        
        # 1. Determine adaptive target sparsity
        target_sparsity = self.pruning_scheduler.get_sparsity_for_cluster(
            cluster_data_labels, num_clients
        )
        print(f"    Target sparsity: {target_sparsity:.2%}")
        
        # 2. Calculate hybrid importance scores
        importance_scores = self.importance_scorer.calculate_scores(
            dense_model, client_models_in_cluster
        )
        print(f"    Calculated importance scores for {len(importance_scores)} layers")
        
        # 3. Create pruned model and apply mask based on scores
        pruned_model = create_lstm_model()
        pruned_model.set_weights(dense_model.get_weights())
        
        for layer in pruned_model.layers:
            if not hasattr(layer, 'kernel'):
                continue
            
            layer_name = layer.name
            if layer_name not in importance_scores:
                continue
            
            weights = layer.kernel.numpy()
            scores = importance_scores[layer_name]
            
            # Flatten for easier processing
            flat_weights = weights.flatten()
            flat_scores = scores.flatten()
            
            # Determine threshold for pruning
            n_weights = len(flat_weights)
            n_prune = int(n_weights * target_sparsity)
            
            # Sort by importance score and prune lowest
            prune_indices = np.argsort(flat_scores)[:n_prune]
            mask = np.ones_like(flat_weights)
            mask[prune_indices] = 0
            
            # Apply mask
            pruned_weights = flat_weights * mask
            layer.kernel.assign(pruned_weights.reshape(weights.shape))
        
        # Store final pruned model
        self.final_pruned_models[cluster_id] = pruned_model
        
        actual_sparsity = get_model_sparsity(pruned_model)
        print(f"    Actual sparsity achieved: {actual_sparsity:.2f}%")
        
        return pruned_model


class CAAFPClient:
    """Client for CA-AFP framework."""
    def __init__(self, client_id, train_dataset, test_dataset, lambda_reg=0.01):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lambda_reg = lambda_reg
        
        self.personal_model = create_lstm_model()
        self.cluster_model = create_lstm_model()
    
    def train_with_regularization(self, cluster_weights, epochs=5):
        """Train with regularization to cluster model."""
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
                                     cluster_weights):
                        diff = pw - cw
                        reg_loss += tf.reduce_sum(tf.square(diff))
                    
                    reg_loss = (self.lambda_reg / 2.0) * reg_loss
                    total_loss = local_loss + reg_loss
                
                gradients = tape.gradient(total_loss, self.personal_model.trainable_weights)
                self.personal_model.optimizer.apply_gradients(
                    zip(gradients, self.personal_model.trainable_weights)
                )
        
        # Also update cluster model
        self.cluster_model.fit(self.train_dataset, epochs=epochs, verbose=0)
        
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


def run_caafp(num_clients=30, num_rounds=100, initial_rounds=10,
              clustering_training_rounds=30, clients_per_round=10,
              epochs_per_round=5, fine_tune_epochs=3):
    """
    Main CA-AFP training loop.
    """
    print("="*60)
    print("CA-AFP: Cluster-Aware Adaptive Federated Pruning")
    print("="*60)
    
    print("\nLoading and preparing data...")
    X_data, y_data, user_ids = load_wisdm_dataset()
    client_data = create_non_iid_data_split(X_data, y_data, user_ids, num_clients=num_clients)
    train_datasets, test_datasets = create_tf_datasets(client_data, batch_size=32)
    
    # Initialize server and clients
    print("Initializing server and clients...")
    server = CAAFPServer(num_clients=num_clients, num_clusters=3)
    clients = {
        i: CAAFPClient(i, train_datasets[i], test_datasets[i], lambda_reg=0.01)
        for i in range(num_clients) if i in train_datasets
    }
    
    print(f"\n{'='*60}")
    print("PHASE 1 & 2: Initial Training and Clustering")
    print(f"{'='*60}")
    
    # Phase 1: Initial FL rounds
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
    
    # Phase 2: Clustering
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
    
    # Phase 3: Train specialized dense models for each cluster
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
    print("PHASE 4: Cluster-Aware Adaptive Pruning")
    print(f"{'='*60}")
    
    # Phase 4: Apply novel pruning to each cluster
    for cluster_id in range(server.num_clusters):
        cluster_client_ids = [
            cid for cid, clust in server.client_clusters.items()
            if clust == cluster_id
        ]
        
        if not cluster_client_ids:
            continue
        
        # Collect client models and data labels
        client_models = [clients[cid].personal_model for cid in cluster_client_ids]
        
        # Collect all labels from clients in this cluster
        cluster_labels_list = []
        for client_id in cluster_client_ids:
            for _, y in clients[client_id].train_dataset:
                cluster_labels_list.append(y.numpy())
        cluster_data_labels = np.concatenate(cluster_labels_list)
        
        # Apply cluster-aware adaptive pruning
        pruned_model = server.apply_cluster_aware_pruning(
            cluster_id, client_models, cluster_data_labels
        )
        
        # Fine-tune the pruned model with cluster clients
        print(f"    Fine-tuning pruned model...")
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
    print(f"Std Dev:          {np.std(accuracies):.4f}")
    print(f"Min:              {np.min(accuracies):.4f}")
    print(f"Max:              {np.max(accuracies):.4f}")
    
    # Per-cluster statistics
    print(f"\nPer-Cluster Statistics:")
    for cluster_id in range(server.num_clusters):
        if cluster_results[cluster_id]:
            cluster_acc = cluster_results[cluster_id]
            cluster_sparsity = get_model_sparsity(
                server.final_pruned_models[cluster_id]
            )
            print(f"  Cluster {cluster_id}:")
            print(f"    Clients:  {len(cluster_acc)}")
            print(f"    Avg Acc:  {np.mean(cluster_acc):.4f}")
            print(f"    Std Dev:  {np.std(cluster_acc):.4f}")
            print(f"    Sparsity: {cluster_sparsity:.2f}%")
    
    return server, clients, results


if __name__ == "__main__":
    server, clients, results = run_caafp(
        num_clients=30,
        num_rounds=100,
        initial_rounds=10,
        clustering_training_rounds=30,
        clients_per_round=10,
        epochs_per_round=3,
        fine_tune_epochs=3
    )