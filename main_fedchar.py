import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
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
            
class FedCHARServer:
    def __init__(self, num_clients, num_clusters=3):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.global_model = create_lstm_model()
        self.cluster_models = {}
        self.client_clusters = {}  # Maps client_id to cluster_id
        self.clustered = False
    
    def aggregate_weights(self, client_weights, client_ids=None):
        """
        FedAvg aggregation: average the weights from all clients.
        """
        if client_ids is None:
            client_ids = list(range(len(client_weights)))
        
        # Initialize with zeros
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        
        # Sum all weights
        for weights in client_weights:
            for i, w in enumerate(weights):
                avg_weights[i] += w
        
        # Average
        for i in range(len(avg_weights)):
            avg_weights[i] /= len(client_weights)
        
        return avg_weights
    
    def cluster_clients(self, client_updates):
        """
        Cluster clients based on cosine similarity of their model updates.
        """
        # Flatten updates for similarity calculation
        flattened_updates = []
        for update in client_updates:
            flattened = np.concatenate([u.flatten() for u in update])
            flattened_updates.append(flattened)
        
        flattened_updates = np.array(flattened_updates)
        
        # Calculate cosine similarity matrix
        n_clients = len(flattened_updates)
        similarity_matrix = np.zeros((n_clients, n_clients))
        
        for i in range(n_clients):
            for j in range(n_clients):
                if i != j:
                    similarity = 1 - cosine(flattened_updates[i], flattened_updates[j])
                    similarity_matrix[i, j] = similarity
                else:
                    similarity_matrix[i, j] = 1.0
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Perform agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=self.num_clusters,
            metric='precomputed',
            linkage='complete'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Store cluster assignments
        for client_id, cluster_id in enumerate(cluster_labels):
            self.client_clusters[client_id] = int(cluster_id)
        
        # Initialize cluster models
        for cluster_id in range(self.num_clusters):
            self.cluster_models[cluster_id] = create_lstm_model()
            self.cluster_models[cluster_id].set_weights(self.global_model.get_weights())
        
        self.clustered = True
        
        return cluster_labels
    
    def get_cluster_weights(self, cluster_id):
        """Get weights for a specific cluster model."""
        if cluster_id in self.cluster_models:
            return self.cluster_models[cluster_id].get_weights()
        return self.global_model.get_weights()
    
    def update_cluster_model(self, cluster_id, client_weights, client_ids):
        """Update the model for a specific cluster."""
        avg_weights = self.aggregate_weights(client_weights, client_ids)
        self.cluster_models[cluster_id].set_weights(avg_weights)


class FedCHARClient:
    def __init__(self, client_id, train_dataset, test_dataset, lambda_reg=0.01):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lambda_reg = lambda_reg
        
        # Two models: personal and group
        self.personal_model = create_lstm_model()
        self.group_model = create_lstm_model()
    
    def train_step(self, x, y, global_weights):
        """
        Single training step with FedCHAR objective.
        Implements: L_total = L_local + (lambda/2) * ||v_k - w_g||^2
        """
        with tf.GradientTape() as tape:
            predictions = self.personal_model(x, training=True)
            local_loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
            local_loss = tf.reduce_mean(local_loss)
            
            # Regularization term: distance between personal and group model
            reg_loss = 0.0
            personal_weights = self.personal_model.trainable_weights
            
            for i, pw in enumerate(personal_weights):
                if i < len(global_weights):
                    diff = pw - global_weights[i]
                    reg_loss += tf.reduce_sum(tf.square(diff))
            
            reg_loss = (self.lambda_reg / 2.0) * reg_loss
            total_loss = local_loss + reg_loss
        
        gradients = tape.gradient(total_loss, self.personal_model.trainable_weights)
        self.personal_model.optimizer.apply_gradients(
            zip(gradients, self.personal_model.trainable_weights)
        )
        
        return local_loss.numpy(), reg_loss.numpy()
    
    def train_fedchar(self, global_weights, epochs=5):
        """Train with FedCHAR objective."""
        # Set group model weights
        self.group_model.set_weights(global_weights)
        
        for epoch in range(epochs):
            for x, y in self.train_dataset:
                local_loss, reg_loss = self.train_step(x, y, global_weights)
        
        # Also train the group model normally (for clustering/aggregation)
        self.group_model.fit(self.train_dataset, epochs=epochs, verbose=0)
        
        return self.group_model.get_weights()
    
    def compute_update(self, old_weights):
        """Compute model update (difference between new and old weights)."""
        new_weights = self.group_model.get_weights()
        update = [new - old for new, old in zip(new_weights, old_weights)]
        return update
    
    def evaluate(self):
        """Evaluate the personal model."""
        results = self.personal_model.evaluate(self.test_dataset, verbose=0)
        return {'loss': results[0], 'accuracy': results[1]}


def run_fedchar(num_clients=30, num_rounds=100, initial_rounds=10, 
                clients_per_round=10, epochs_per_round=5):
    """
    Main FedCHAR training loop.
    """
    print("Loading and preparing data...")
    X_data, y_data, user_ids = load_wisdm_dataset()
    client_data = create_non_iid_data_split(X_data, y_data, user_ids, num_clients=num_clients)
    train_datasets, test_datasets = create_tf_datasets(client_data, batch_size=32)
    
    # Initialize server and clients
    print("Initializing server and clients...")
    server = FedCHARServer(num_clients=num_clients, num_clusters=3)
    clients = {
        i: FedCHARClient(i, train_datasets[i], test_datasets[i], lambda_reg=0.01)
        for i in range(num_clients) if i in train_datasets
    }
    
    print(f"\nStarting FedCHAR training for {num_rounds} rounds...")
    
    # Phase 1: Initial FL rounds (before clustering)
    print(f"\n--- Phase 1: Initial Training ({initial_rounds} rounds) ---")
    for round_num in range(initial_rounds):
        print(f"Round {round_num + 1}/{initial_rounds}")
        
        # Select random clients
        selected_clients = np.random.choice(
            list(clients.keys()),
            size=min(clients_per_round, len(clients)),
            replace=False
        )
        
        # Get global weights
        global_weights = server.global_model.get_weights()
        
        # Train clients
        client_weights = []
        for client_id in selected_clients:
            weights = clients[client_id].train_fedchar(global_weights, epochs=epochs_per_round)
            client_weights.append(weights)
        
        # Aggregate
        avg_weights = server.aggregate_weights(client_weights)
        server.global_model.set_weights(avg_weights)
    
    # Phase 2: Clustering
    print(f"\n--- Phase 2: Client Clustering ---")
    global_weights = server.global_model.get_weights()
    
    # Collect updates from all clients
    client_updates = []
    for client_id in clients.keys():
        # Train for one round
        clients[client_id].group_model.set_weights(global_weights)
        clients[client_id].group_model.fit(clients[client_id].train_dataset, epochs=1, verbose=0)
        update = clients[client_id].compute_update(global_weights)
        client_updates.append(update)
    
    # Perform clustering
    cluster_labels = server.cluster_clients(client_updates)
    print(f"Clustering complete. Cluster assignments: {server.client_clusters}")
    
    # Phase 3: Personalized training within clusters
    print(f"\n--- Phase 3: Personalized Training ({num_rounds - initial_rounds} rounds) ---")
    for round_num in range(initial_rounds, num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        
        # Train within each cluster
        for cluster_id in range(server.num_clusters):
            # Get clients in this cluster
            cluster_client_ids = [
                cid for cid, clust in server.client_clusters.items()
                if clust == cluster_id
            ]
            
            if not cluster_client_ids:
                continue
            
            # Select subset of cluster clients
            selected = np.random.choice(
                cluster_client_ids,
                size=min(clients_per_round // server.num_clusters + 1, len(cluster_client_ids)),
                replace=False
            )
            
            # Get cluster model weights
            cluster_weights = server.get_cluster_weights(cluster_id)
            
            # Train clients
            client_weights = []
            for client_id in selected:
                weights = clients[client_id].train_fedchar(cluster_weights, epochs=epochs_per_round)
                client_weights.append(weights)
            
            # Update cluster model
            server.update_cluster_model(cluster_id, client_weights, selected)
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    results = {}
    for client_id, client in clients.items():
        eval_results = client.evaluate()
        results[client_id] = eval_results
        cluster_id = server.client_clusters[client_id]
        print(f"Client {client_id} (Cluster {cluster_id}): "
              f"Accuracy = {eval_results['accuracy']:.4f}")
    
    # Calculate statistics
    accuracies = [r['accuracy'] for r in results.values()]
    print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Dev: {np.std(accuracies):.4f}")
    print(f"Min: {np.min(accuracies):.4f}, Max: {np.max(accuracies):.4f}")
    
    return server, clients, results


if __name__ == "__main__":
    # --- Setup logging and paths ---
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    log_filepath = f'logs/run_log_fedchar_{run_id}.log'
    model_save_path = f'saved_models/model_fedchar_{run_id}'
    
    original_stdout = sys.stdout
    logger = Logger(log_filepath)
    sys.stdout = logger

    print(f"--- Starting FedCHAR Run: {run_id} ---")
    print(f"Logs will be saved to: {log_filepath}")
    print(f"Models will be saved to: {model_save_path}_*.h5")

    try:
        # --- Run the experiment ---
        server, clients, results = run_fedchar(
            num_clients=30,
            num_rounds=50,
            initial_rounds=10,
            clients_per_round=10,
            epochs_per_round=3
        )
        
        # --- Save the models ---
        print(f"\n--- Saving models for run {run_id} ---")
        # FedCHAR saves the dense cluster models
        for cluster_id, model in server.cluster_models.items():
            save_path = f"{model_save_path}_cluster_{cluster_id}.h5"
            model.save_weights(save_path)
            print(f"Saved model for cluster {cluster_id} to {save_path}")

    finally:
        # --- Restore stdout and close log ---
        sys.stdout = original_stdout
        del logger
        print(f"Run {run_id} complete. Logs saved to {log_filepath}")