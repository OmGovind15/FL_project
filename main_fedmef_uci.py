import os
# Use GPU 0 and disable XLA/oneDNN for compatibility/reproducibility
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set log level to '2' to suppress most TensorFlow warnings/info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import sys
import os
from datetime import datetime
from data_loader_uci import load_uci_har_dataset, create_non_iid_data_split, create_tf_datasets
from models import create_prunable_lstm_model, get_model_sparsity

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

class FedMefServer:
    def __init__(self, num_clients, target_sparsity=0.8):
        self.num_clients = num_clients
        self.target_sparsity = target_sparsity
        self.global_model = None
        self.global_mask = {}  # Stores which weights are active

    def initialize_model(self, input_shape=(128, 9), num_classes=6):
        """Initialize a randomly pruned model."""
        self.global_model = create_prunable_lstm_model(
            input_shape=input_shape,
            num_classes=num_classes,
            final_sparsity=self.target_sparsity
        )
        
        # Apply random initial pruning
        self._apply_random_pruning()

    def _apply_random_pruning(self):
        """Apply random pruning to achieve target sparsity."""
        for layer in self.global_model.layers:
            # Check for a standard layer with weights
            if hasattr(layer, 'kernel'):
                weights = layer.kernel.numpy()
                flat_weights = weights.flatten()
                n_prune = int(len(flat_weights) * self.target_sparsity)
                
                # Randomly select weights to prune
                prune_indices = np.random.choice(len(flat_weights), n_prune, replace=False)
                mask = np.ones_like(flat_weights)
                mask[prune_indices] = 0
                
                # Apply mask
                masked_weights = flat_weights * mask
                layer.kernel.assign(masked_weights.reshape(weights.shape))
                
                # Store mask
                self.global_mask[layer.name] = mask.reshape(weights.shape)

    def aggregate_weights(self, client_weights):
        """Aggregate weights from clients."""
        if not client_weights:  # FIXED: guard against empty list
            return self.global_model.get_weights()
        
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        for weights in client_weights:
            for i, w in enumerate(weights):
                avg_weights[i] += w
        for i in range(len(avg_weights)):
            avg_weights[i] /= len(client_weights)
        return avg_weights

    def aggregate_gradients(self, client_gradients):
        """Aggregate gradients from clients."""
        if not client_gradients:  # FIXED: guard against empty list
            return {}
        
        avg_grads = {}
        for layer_name in client_gradients[0].keys():
            grads = [cg[layer_name] for cg in client_gradients]
            avg_grads[layer_name] = np.mean(grads, axis=0)
        return avg_grads

    def adjust_masks(self, avg_gradients, prune_rate=0.2):
        """
        Adjust masks: prune lowest-magnitude active weights, grow highest-gradient inactive weights.
        Per FedMef paper Algorithm 1.
        """
        for layer in self.global_model.layers:
            # Correctly check for a standard layer with weights
            if hasattr(layer, 'kernel'):
                # This code will now execute correctly
                layer_name = layer.name
                weights = layer.kernel.numpy()
                
                # Get current mask
                if layer_name not in self.global_mask:
                    self.global_mask[layer_name] = np.ones_like(weights)
                
                current_mask = self.global_mask[layer_name]
                
                # Calculate number of weights to prune/grow
                active_weights_count = np.sum(current_mask > 0)
                n_adjust = int(active_weights_count * prune_rate)
                
                # PRUNE: Remove lowest magnitude active weights
                active_weights_flat = weights.flatten() * current_mask.flatten()
                active_indices = np.where(current_mask.flatten() > 0)[0]
                
                if len(active_indices) > n_adjust:
                    active_magnitudes = np.abs(active_weights_flat[active_indices])
                    prune_indices = active_indices[np.argsort(active_magnitudes)[:n_adjust]]
                    
                    # Update mask
                    new_mask = current_mask.flatten()
                    new_mask[prune_indices] = 0
                    
                    # GROW: Enable highest gradient magnitude inactive weights
                    if layer_name in avg_gradients:
                        inactive_indices = np.where(new_mask == 0)[0]
                        
                        if len(inactive_indices) > 0:
                            gradient_magnitudes = np.abs(avg_gradients[layer_name].flatten())
                            inactive_grad_magnitudes = gradient_magnitudes[inactive_indices]
                            
                            # Ensure we don't grow more than available inactive spots
                            num_to_grow = min(n_adjust, len(inactive_indices))
                            grow_indices = inactive_indices[
                                np.argsort(inactive_grad_magnitudes)[-num_to_grow:]
                            ]
                            new_mask[grow_indices] = 1
                    
                    # Apply new mask
                    new_mask = new_mask.reshape(weights.shape)
                    masked_weights = weights * new_mask
                    layer.kernel.assign(masked_weights)
                    self.global_mask[layer.name] = new_mask

    def apply_mask(self):
        """Enforce current global mask on model weights."""
        for layer in self.global_model.layers:
            if hasattr(layer, 'kernel') and layer.name in self.global_mask:
                weights = layer.kernel.numpy()
                mask = self.global_mask[layer.name]
                layer.kernel.assign(weights * mask)


class FedMefClient:
    def __init__(self, client_id, train_dataset, test_dataset):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = None
        self.bae_lambda = 0.01  # Regularization coefficient for BaE

    def set_model(self, global_model):
        """Receive model from server."""
        self.model = tf.keras.models.clone_model(global_model)
        self.model.set_weights(global_model.get_weights())
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def standard_training(self, epochs=5):
        """Standard training on sparse model."""
        self.model.fit(
            self.train_dataset,
            epochs=epochs,
            verbose=0
        )
        return self.model.get_weights()

    def bae_training(self, epochs=3, low_magnitude_percentile=20):
        """
        Budget-Aware Extrusion (BaE) training.
        Adds penalty to low-magnitude weights to transfer their information.
        """
        # Identify low-magnitude weights
        low_mag_masks = {}
        for layer in self.model.layers:
            # Check for a standard layer with weights
            if hasattr(layer, 'kernel'):
                weights = layer.kernel.numpy()
                
                # Ensure there are non-zero weights before calling percentile
                non_zero_weights = weights[weights != 0]
                if non_zero_weights.size > 0:
                    threshold = np.percentile(np.abs(non_zero_weights), low_magnitude_percentile)
                    low_mag_mask = (np.abs(weights) < threshold) & (weights != 0)
                    low_mag_masks[layer.name] = low_mag_mask

        # Training with BaE loss
        for epoch in range(epochs):
            for x, y in self.train_dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(x, training=True)
                    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
                    ce_loss = tf.reduce_mean(ce_loss)
                    
                    # BaE regularization: penalize low-magnitude weights
                    reg_loss = 0.0
                    for layer in self.model.layers:
                        # Check for a standard layer with weights
                        if hasattr(layer, 'kernel'):
                            if layer.name in low_mag_masks:
                                weights = layer.kernel
                                mask_tf = tf.constant(low_mag_masks[layer.name], dtype=tf.float32)
                                low_mag_weights = weights * mask_tf
                                reg_loss += tf.reduce_sum(tf.square(low_mag_weights))
                    
                    reg_loss = self.bae_lambda * reg_loss
                    total_loss = ce_loss + reg_loss
                
                gradients = tape.gradient(total_loss, self.model.trainable_weights)
                self.model.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_weights)
                )
        
        return self.model.get_weights()

    def compute_gradients(self):
        """Compute and return gradients for mask adjustment."""
        gradients = {}
        
        # Take a small batch to compute gradients
        for x, y in self.train_dataset.take(1):
            with tf.GradientTape() as tape:
                predictions = self.model(x, training=False)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
                loss = tf.reduce_mean(loss)
            
            # Match trainable weights to layers correctly
            trainable_weights_map = {var.name: var for var in self.model.trainable_weights}
            grads = tape.gradient(loss, list(trainable_weights_map.values()))
            
            # Store gradients by layer name, matching weights to grads
            grad_map = {var.name: grad for var, grad in zip(trainable_weights_map.values(), grads)}
            
            for layer in self.model.layers:
                # Check for a standard layer with weights
                if hasattr(layer, 'kernel'):
                    # The kernel's name is typically 'layer_name/kernel:0'
                    kernel_name = f"{layer.name}/kernel:0"
                    if kernel_name in grad_map and grad_map[kernel_name] is not None:
                        gradients[layer.name] = grad_map[kernel_name].numpy()
        
        return gradients

    def evaluate(self):
        """Evaluate the model."""
        results = self.model.evaluate(self.test_dataset, verbose=0)
        return {'loss': results[0], 'accuracy': results[1]}


def run_fedmef(num_clients=30, num_rounds=100, adjustment_interval=10,
               stop_adjustment_round=60, clients_per_round=10,
               epochs_per_round=5, target_sparsity=0.8):
    """
    Main FedMef training loop.
    """
    print("Loading and preparing data...")
    X_data, y_data, user_ids = load_uci_har_dataset()
    client_data = create_non_iid_data_split(X_data, y_data, user_ids, num_clients=num_clients)
    train_datasets, test_datasets = create_tf_datasets(client_data, batch_size=32)

    # Initialize server and clients
    print("Initializing server and clients...")
    server = FedMefServer(num_clients=num_clients, target_sparsity=target_sparsity)
    server.initialize_model(input_shape=(128, 9), num_classes=6)
    
    clients = {
        i: FedMefClient(i, train_datasets[i], test_datasets[i])
        for i in range(num_clients) if i in train_datasets
    }

    print(f"\nStarting FedMef training for {num_rounds} rounds...")
    print(f"Initial sparsity: {get_model_sparsity(server.global_model):.2%}")

    for round_num in range(num_rounds):
        is_adjustment_round = (round_num > 0 and 
                             round_num % adjustment_interval == 0 and 
                             round_num <= stop_adjustment_round)
        
        print(f"\nRound {round_num + 1}/{num_rounds} "
              f"{'(Adjustment Round)' if is_adjustment_round else ''}")

        # Select random clients
        selected_clients = np.random.choice(
            list(clients.keys()),
            size=min(clients_per_round, len(clients)),
            replace=False
        )

        # Send model to clients
        for client_id in selected_clients:
            clients[client_id].set_model(server.global_model)

        # Train clients
        client_weights = []
        client_gradients = []
        
        for client_id in selected_clients:
            if is_adjustment_round:
                # Use BaE training in adjustment rounds
                weights = clients[client_id].bae_training(epochs=epochs_per_round)
                gradients = clients[client_id].compute_gradients()
                client_gradients.append(gradients)
                client_weights.append(weights)  # FIXED: append weights in adjustment rounds
            else:
                # Standard training
                weights = clients[client_id].standard_training(epochs=epochs_per_round)
                client_weights.append(weights)

        # Server aggregates weights every round (FIXED)
        if client_weights:
            avg_weights = server.aggregate_weights(client_weights)
            server.global_model.set_weights(avg_weights)

        # Apply mask every round (FIXED: moved outside conditional)
        server.apply_mask()

        # Adjust masks if needed (only on adjustment rounds)
        if is_adjustment_round and client_gradients:
            print("  Adjusting masks (pruning and growing)...")
            avg_gradients = server.aggregate_gradients(client_gradients)
            server.adjust_masks(avg_gradients, prune_rate=0.2)
            # Re-apply mask after adjustment
            server.apply_mask()

        # Print sparsity
        if round_num % 10 == 0 or is_adjustment_round:
            sparsity = get_model_sparsity(server.global_model)
            print(f"  Current sparsity: {sparsity:.2%}")

    # Final evaluation
    print("\n--- Final Evaluation ---")
    results = {}
    for client_id, client in clients.items():
        client.set_model(server.global_model)
        eval_results = client.evaluate()
        results[client_id] = eval_results
        print(f"Client {client_id}: Accuracy = {eval_results['accuracy']:.4f}")

    # Calculate statistics
    accuracies = [r['accuracy'] for r in results.values()]
    final_sparsity = get_model_sparsity(server.global_model)
    
    print(f"\nFinal Sparsity: {final_sparsity:.2%}")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Dev: {np.std(accuracies):.4f}")
    print(f"Min: {np.min(accuracies):.4f}, Max: {np.max(accuracies):.4f}")

    return server, clients, results


if __name__ == "__main__":
    import sys
    from datetime import datetime
    import argparse  # <-- ADD THIS
    import random    # <-- ADD THIS
    import os        # <-- ADD THIS
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
# --- Setup logging and paths ---
    run_id = f"fedmef_seed_{SEED}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs('logs', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    log_filepath = f'logs/run_log_{run_id}.log'
    model_save_path = f'saved_models/model_{run_id}'
    
    original_stdout = sys.stdout
    logger = Logger(log_filepath)
    sys.stdout = logger
    
    print(f"--- Starting FedMef Run: {run_id} with seed {SEED} ---")
    print(f"Logs will be saved to: {log_filepath}")
    print(f"Models will be saved to: {model_save_path}_*.h5")
    
    try:
        # --- Run the experiment ---
        server, clients, results = run_fedmef(
            num_clients=30,
            num_rounds=50,
            adjustment_interval=10,
            stop_adjustment_round=30,
            clients_per_round=10,
            epochs_per_round=3,
            target_sparsity=0.5
        )
        
        # --- Save the model ---
        print(f"\n--- Saving model for run {run_id} ---")
        save_path = f"{model_save_path}_global_model.h5"
        server.global_model.save_weights(save_path)
        print(f"Saved global model to {save_path}")
        
    finally:
        # --- Restore stdout and close log ---
        sys.stdout = original_stdout
        del logger
        print(f"Run {run_id} complete. Logs saved to {log_filepath}")
