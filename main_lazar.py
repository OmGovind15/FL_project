"""
Lazarevich et al. Post-Training Pruning Baseline
=================================================
Paper: "Post-training deep neural network pruning via layer-wise calibration" (ICCV 2021)
Authors: Lazarevich, Kozlov, Malinin (Intel)

Implementation of the Lazarevich et al. (2021) post-training pruning
method as a baseline for fair comparison against CA-AFP.

Fair Comparison Setup:
- Same model architecture (LSTM-based HAR)
- Same dataset splits (WISDM with your non-IID distribution)
- Same FL parameters (30 clients, 10 clients/round, 3 local epochs)
- Same total training budget (40 rounds dense training)
- Fine-tuning: 3 epochs of local fine-tuning (to match CA-AFP)
- Calibration: 1 epoch of layer-wise MSE fine-tuning (as per paper)

Key Methodology from Paper:
1. Train a dense model via standard FedAvg.
2. Apply a progressive (cubic) pruning schedule.
3. In each pruning step:
    a. Apply layer-wise L2-normalized magnitude pruning (global threshold)
    b. Apply bias & variance correction for weights and activations
    c. Apply layer-wise fine-tuning with MSE loss (knowledge distillation)
4. After all pruning steps, perform final local fine-tuning for fair comparison.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
from scipy.stats import entropy # Included for consistency with final_2.py
import copy
import sys
import os
from datetime import datetime
# Import from your project files
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

class PostTrainPruningServer:
    """Server for post-training pruning approach."""
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.global_model = create_lstm_model()
        
    def aggregate_weights(self, client_weights):
        """FedAvg aggregation."""
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        
        for weights in client_weights:
            for i, w in enumerate(weights):
                avg_weights[i] += w
        
        for i in range(len(avg_weights)):
            avg_weights[i] /= len(client_weights)
        
        return avg_weights


class PostTrainPruningClient:
    """Client for post-training pruning."""
    def __init__(self, client_id, train_dataset, test_dataset):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = create_lstm_model()
    
    def train(self, global_weights, epochs=3):
        """Standard training or fine-tuning."""
        self.model.set_weights(global_weights)
        self.model.fit(self.train_dataset, epochs=epochs, verbose=0)
        return self.model.get_weights()
    
    def evaluate(self):
        """Evaluate model."""
        results = self.model.evaluate(self.test_dataset, verbose=0)
        return {'loss': results[0], 'accuracy': results[1]}
    
    def set_model_weights(self, weights):
        """Set model weights."""
        self.model.set_weights(weights)


def apply_layerwise_l2_normalized_pruning(model, target_sparsity):
    """
    Apply L2-normalized magnitude pruning (Lazarevich et al. method).
    
    Key insight from paper: L2-normalized magnitude performs better than
    raw magnitude when BatchNorm is fused or when comparing across layers.
    
    Formula: I(w_i^l) = |w_i^l| / sqrt(sum_j(|w_j^l|^2)) (Eq. 2)
    """
    print(f"    Applying L2-normalized magnitude pruning...")
    
    # Collect all importance scores across all layers
    all_scores = []
    layer_info = []
    
    for layer in model.layers:
        if not hasattr(layer, 'kernel'):
            continue
        
        weights = layer.kernel.numpy()
        
        # Calculate L2-normalized magnitude (Equation 2 from paper)
        weight_magnitudes = np.abs(weights)
        l2_norm = np.sqrt(np.sum(weight_magnitudes ** 2))
        
        if l2_norm > 1e-9:
            normalized_scores = weight_magnitudes / l2_norm
        else:
            normalized_scores = weight_magnitudes
        
        flat_scores = normalized_scores.flatten()
        all_scores.extend(flat_scores)
        layer_info.append({
            'layer': layer,
            'weights': weights,
            'scores': normalized_scores,
            'n_weights': len(flat_scores)
        })
    
    # Find global threshold based on target sparsity
    all_scores = np.array(all_scores)
    n_total_weights = len(all_scores)
    n_prune = int(n_total_weights * target_sparsity)
    
    # Ensure n_prune is within valid range
    n_prune = max(0, min(n_prune, n_total_weights - 1))
    
    if n_total_weights > 0:
        threshold = np.sort(all_scores)[n_prune]
    else:
        threshold = 0.0
    
    print(f"    Global threshold: {threshold:.6f}")
    
    # Apply pruning to each layer
    for info in layer_info:
        layer = info['layer']
        weights = info['weights']
        scores = info['scores']
        
        # Create binary mask
        mask = (scores >= threshold).astype(np.float32)
        pruned_weights = weights * mask
        
        layer.kernel.assign(pruned_weights)
    
    actual_sparsity = get_model_sparsity(model)
    print(f"    Actual sparsity achieved: {actual_sparsity:.2f}%")
    
    return model


def apply_bias_variance_correction(model, dense_model, calibration_data):
    """
    Apply bias and variance correction (Section 3.2 from paper).
    
    Two corrections:
    1. Weight bias/variance correction (Equations 3-4)
    2. Activation bias correction (Equation 5)
    """
    print("    Applying bias & variance correction...")
    
    try:
        x_batch, _ = next(iter(calibration_data))
    except StopIteration:
        print("    WARNING: Calibration data is empty. Skipping bias correction.")
        return model

    # Store dense model activations (outputs) for bias correction
    dense_activations = {}
    
    output_layers = [layer.output for layer in dense_model.layers if hasattr(layer, 'kernel')]
    intermediate_model = tf.keras.Model(
        inputs=dense_model.input,
        outputs=output_layers
    )
    dense_outputs = intermediate_model(x_batch, training=False)
    
    if not isinstance(dense_outputs, list):
        dense_outputs = [dense_outputs]
        
    layer_idx = 0
    for layer in dense_model.layers:
        if hasattr(layer, 'kernel'):
            dense_activations[layer.name] = dense_outputs[layer_idx].numpy()
            layer_idx += 1
    
    # Apply corrections layer by layer
    for sparse_layer, dense_layer in zip(model.layers, dense_model.layers):
        if not hasattr(sparse_layer, 'kernel'):
            continue
        
        W_sparse = sparse_layer.kernel.numpy()
        W_dense = dense_layer.kernel.numpy()
        
        # *** FIX: Get the mask BEFORE correction ***
        mask = (W_sparse != 0).astype(np.float32)
        
        # 1. Weight variance and bias correction (Equations 3-4)
        n_channels = W_sparse.shape[-1]
        
        for ch in range(n_channels):
            if len(W_sparse.shape) == 2:  # Dense layer
                w_sparse_ch = W_sparse[:, ch]
                w_dense_ch = W_dense[:, ch]
                mask_ch = mask[:, ch] # Get channel mask
            else:  # Conv/LSTM layer
                w_sparse_ch = W_sparse[..., ch].flatten()
                w_dense_ch = W_dense[..., ch].flatten()
                mask_ch = mask[..., ch].flatten() # Get channel mask
            
            # Calculate statistics (using all weights, as per paper)
            mean_dense = np.mean(w_dense_ch)
            std_dense = np.std(w_dense_ch)
            mean_sparse = np.mean(w_sparse_ch)
            std_sparse = np.std(w_sparse_ch)
            
            lambda_scale = 1.0
            if std_sparse > 1e-9:
                lambda_scale = std_dense / (std_sparse + 1e-9)
            
            # Apply scaling and bias correction
            corrected_w_ch = lambda_scale * w_sparse_ch + (mean_dense - lambda_scale * mean_sparse)
            
            # *** FIX: Re-apply the mask AFTER correction ***
            # This ensures pruned weights remain zero
            corrected_w_ch = corrected_w_ch * mask_ch
            
            if len(W_sparse.shape) == 2:
                W_sparse[:, ch] = corrected_w_ch
            else:
                W_sparse[..., ch] = corrected_w_ch.reshape(W_sparse[..., ch].shape)
        
        sparse_layer.kernel.assign(W_sparse)
        
        # 2. Activation bias correction (this part was fine)
        if hasattr(sparse_layer, 'bias') and sparse_layer.bias is not None:
            sparse_intermediate = tf.keras.Model(
                inputs=model.input,
                outputs=sparse_layer.output
            )
            sparse_act = sparse_intermediate(x_batch, training=False).numpy()
            
            if sparse_layer.name in dense_activations:
                dense_act = dense_activations[sparse_layer.name]
                bias_shift = np.mean(dense_act - sparse_act)
                current_bias = sparse_layer.bias.numpy()
                sparse_layer.bias.assign(current_bias + bias_shift)
    
    print("    Bias & variance correction complete")
    return model


def layerwise_finetuning(model, dense_model, calibration_data, epochs=1, learning_rate=1e-5):
    """
    Layer-wise fine-tuning with MSE loss (Section 3.3 from paper).
    
    Key points from paper:
    - Use MSE loss between sparse and dense layer outputs
    - Optimize each layer independently
    - Adam optimizer with lr=1e-5 for weights
    - No weight decay (paper found it hurts performance)
    """
    print(f"    Applying layer-wise fine-tuning ({epochs} epochs)...")
    
    # Fine-tune each prunable layer
    for layer_idx, (sparse_layer, dense_layer) in enumerate(zip(model.layers, dense_model.layers)):
        if not hasattr(sparse_layer, 'kernel'):
            continue
        
        print(f"      Fine-tuning layer: {sparse_layer.name}")
        
        # Create intermediate models to get layer outputs
        sparse_intermediate = tf.keras.Model(
            inputs=model.input,
            outputs=sparse_layer.output
        )
        dense_intermediate = tf.keras.Model(
            inputs=dense_model.input,
            outputs=dense_layer.output
        )
        
        # Optimizer for this layer only
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Get pruning mask (and keep it fixed)
        weights = sparse_layer.kernel.numpy()
        mask = (weights != 0).astype(np.float32)
        
        # Fine-tune for specified epochs
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for x_batch, _ in calibration_data:
                with tf.GradientTape() as tape:
                    # Get outputs from both models
                    dense_output = dense_intermediate(x_batch, training=False)
                    sparse_output = sparse_intermediate(x_batch, training=True)
                    
                    # MSE loss (Equation 6 from paper)
                    mse_loss = tf.reduce_mean(tf.square(dense_output - sparse_output))
                
                # Compute gradients only for this layer's trainable weights
                gradients = tape.gradient(mse_loss, sparse_layer.trainable_weights)
                
                # --- START OF COMPLETED CODE ---
                
                if gradients:
                    processed_grads = []
                    # sparse_layer.trainable_weights is typically [kernel, bias]
                    for grad, var in zip(gradients, sparse_layer.trainable_weights):
                        if 'kernel' in var.name:
                            # Apply the fixed mask to the kernel's gradient
                            processed_grads.append(grad * mask)
                        else:
                            # Bias gradient is not pruned
                            processed_grads.append(grad)
                    
                    # Apply the masked gradients
                    optimizer.apply_gradients(zip(processed_grads, sparse_layer.trainable_weights))
                
                epoch_loss += mse_loss.numpy()
                n_batches += 1
            
            if n_batches > 0:
                avg_loss = epoch_loss / n_batches
                if epoch == 0 or (epoch + 1) == epochs: # Log first and last
                    print(f"        Epoch {epoch + 1}/{epochs}, MSE Loss: {avg_loss:.6f}")
                
                # --- END OF COMPLETED CODE ---
            
    return model


def run_lazarevich_baseline(num_clients=30, total_dense_rounds=40,
                            clients_per_round=10, epochs_per_round=3,
                            target_sparsity=0.7, pruning_steps=10,
                            calibration_epochs=1, fine_tune_epochs=3):
    """
    Main Lazarevich et al. (2021) baseline training loop.
    
    Args:
        num_clients: Total number of clients
        total_dense_rounds: Rounds for training the initial dense model
        clients_per_round: Clients selected per round
        epochs_per_round: Local epochs for dense training
        target_sparsity: Final sparsity target (e.g., 0.7 for 70%)
        pruning_steps: Iterations for the progressive schedule (paper used 10)
        calibration_epochs: Epochs for *internal* layer-wise MSE fine-tuning
        fine_tune_epochs: Epochs for *final* local fine-tuning (for fair comparison)
    """
    print("="*60)
    print(" Baseline: Lazarevich et al. (2021) Post-Training Pruning")
    print("="*60)
    
    # 1. Load and prepare data
    print("\nLoading and preparing data...")
    X_data, y_data, user_ids = load_wisdm_dataset()
    client_data = create_non_iid_data_split(X_data, y_data, user_ids, num_clients=num_clients)
    train_datasets, test_datasets = create_tf_datasets(client_data, batch_size=32)
    
    # 2. Initialize server and clients
    print("Initializing server and clients...")
    server = PostTrainPruningServer(num_clients=num_clients)
    clients = {
        i: PostTrainPruningClient(i, train_datasets[i], test_datasets[i])
        for i in range(num_clients) if i in train_datasets
    }
    
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training Dense Model ({total_dense_rounds} rounds)")
    print(f"{'='*60}")
    
    for round_num in range(total_dense_rounds):
        print(f"Round {round_num + 1}/{total_dense_rounds}")
        
        selected_clients = np.random.choice(
            list(clients.keys()),
            size=min(clients_per_round, len(clients)),
            replace=False
        )
        
        global_weights = server.global_model.get_weights()
        
        client_weights_list = []
        for client_id in selected_clients:
            weights = clients[client_id].train(
                global_weights, epochs=epochs_per_round
            )
            client_weights_list.append(weights)
        
        avg_weights = server.aggregate_weights(client_weights_list)
        server.global_model.set_weights(avg_weights)
    
    # Store the final dense model
    dense_model = create_lstm_model()
    dense_model.set_weights(server.global_model.get_weights())
    
    print(f"\n{'='*60}")
    print("PHASE 2: Post-Training Pruning & Calibration")
    print(f"{'='*60}")
    
    #
    # === UPDATED CODE BLOCK ===
    # 3. Create "Data-Free" (White Noise) Calibration Set
    # We generate synthetic noise with the same shape as our data
    # (Batch Size = 32, Timesteps = 200, Features = 3)
    # The paper used a few hundred samples; 6 batches = 192 samples.
    CAL_BATCHES = 6
    BATCH_SIZE = 32
    INPUT_SHAPE = (200, 3) 
    NUM_CLASSES = 6 # Must match model output

    # Generate random noise for X (centered at 0)
    cal_x = np.random.normal(size=(CAL_BATCHES * BATCH_SIZE, *INPUT_SHAPE)).astype(np.float32)
    
    # Generate random (but valid) labels for Y
    # The MSE loss doesn't use Y, but the model needs a valid target
    cal_y = np.random.randint(0, NUM_CLASSES, size=(CAL_BATCHES * BATCH_SIZE)).astype(np.int32)
    
    calibration_dataset = tf.data.Dataset.from_tensor_slices((cal_x, cal_y)).batch(BATCH_SIZE)
    print(f"Created 'Data-Free' (White Noise) calibration dataset with {len(cal_x)} samples.")
    # === END OF UPDATED CODE BLOCK ===
    #

    # 4. Apply progressive pruning schedule
    pruned_model = create_lstm_model()
    pruned_model.set_weights(dense_model.get_weights())
    
    s_i = 0.0  # Initial sparsity
    s_f = target_sparsity # Final sparsity
    T = pruning_steps
    
    for t in range(1, T + 1):
        # Cubic schedule (Equation 1 from paper)
        current_sparsity = s_f + (s_i - s_f) * (1 - t / T)**3
        print(f"\nPruning Step {t}/{T}, Target Sparsity: {current_sparsity:.2%}")
        
        # Step 2a: Pruning
        pruned_model = apply_layerwise_l2_normalized_pruning(
            pruned_model, current_sparsity
        )
        
        # Step 2b: Bias Correction
        pruned_model = apply_bias_variance_correction(
            pruned_model, dense_model, calibration_dataset
        )
        
        # Step 2c: Layer-wise Fine-tuning (Calibration)
        pruned_model = layerwise_finetuning(
            pruned_model, dense_model, calibration_dataset, 
            epochs=calibration_epochs
        )
        
    final_pruned_weights = pruned_model.get_weights()
    print("\nProgressive pruning and calibration complete.")
    
    print(f"\n{'='*60}")
    print("PHASE 3: Final Evaluation")
    print(f"{'='*60}")
    
    # 5. Evaluate in two modes for fair comparison
    
    # Mode 1: 0-Shot (Evaluate the pruned global model directly)
    print("\nEvaluating (0-Shot)...")
    results_0_shot = {}
    for client_id, client in clients.items():
        client.set_model_weights(final_pruned_weights)
        eval_results = client.evaluate()
        results_0_shot[client_id] = eval_results
    
    accuracies_0_shot = [r['accuracy'] for r in results_0_shot.values()]
    print(f"  Avg Accuracy (0-Shot): {np.mean(accuracies_0_shot):.4f}")
    
    # Mode 2: Fine-Tuned (Match CA-AFP's 3-epoch local fine-tuning)
    print(f"\nEvaluating ({fine_tune_epochs}-Epoch Local FT)...")
    results_ft = {}
    for client_id, client in clients.items():
        # Each client fine-tunes the pruned model locally
        client.train(final_pruned_weights, epochs=fine_tune_epochs)
        eval_results = client.evaluate()
        results_ft[client_id] = eval_results

    accuracies_ft = [r['accuracy'] for r in results_ft.values()]
    
    # --- Final Results ---
    print(f"\n{'='*60}")
    print("Overall Statistics (Lazarevich et al. Baseline)")
    print("="*60)
    
    print(f"Final Sparsity: {get_model_sparsity(pruned_model):.2f}%")
    
    print("\n--- 0-Shot (No local fine-tuning) ---")
    print(f"Average Accuracy: {np.mean(accuracies_0_shot):.4f}")
    print(f"Std Dev:          {np.std(accuracies_0_shot):.4f}")
    print(f"Min:              {np.min(accuracies_0_shot):.4f}")
    print(f"Max:              {np.max(accuracies_0_shot):.4f}")
    
    print(f"\n--- {fine_tune_epochs}-Epoch Local Fine-Tuning ---")
    print(f"Average Accuracy: {np.mean(accuracies_ft):.4f}")
    print(f"Std Dev:          {np.std(accuracies_ft):.4f}")
    print(f"Min:              {np.min(accuracies_ft):.4f}")
    print(f"Max:              {np.max(accuracies_ft):.4f}")
    
    return results_0_shot, results_ft, pruned_model

if __name__ == "__main__":
    # --- Setup logging and paths ---
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    log_filepath = f'logs/run_log_lazar_{run_id}.log'
    model_save_path = f'saved_models/model_lazar_{run_id}'
    
    original_stdout = sys.stdout
    logger = Logger(log_filepath)
    sys.stdout = logger

    print(f"--- Starting Lazarevich et al. Run: {run_id} ---")
    print(f"Logs will be saved to: {log_filepath}")
    print(f"Models will be saved to: {model_save_path}_*.h5")

    try:
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # --- Run the experiment ---
        # (This now returns the model, thanks to our change in Step 2)
        results_0_shot, results_ft, final_pruned_model = run_lazarevich_baseline(
            num_clients=30,
            total_dense_rounds=40,
            clients_per_round=10,
            epochs_per_round=3,
            target_sparsity=0.5,  # Match CA-AFP base_sparsity
            pruning_steps=10,     # From paper (Table 6)
            calibration_epochs=1, # Internal calibration epochs (hyperparameter)
            fine_tune_epochs=3    # Final local FT (for fair comparison)
        )
        
        # --- Save the model ---
        print(f"\n--- Saving model for run {run_id} ---")
        save_path = f"{model_save_path}_global_pruned.h5"
        final_pruned_model.save_weights(save_path)
        print(f"Saved final pruned model to {save_path}")

    finally:
        # --- Restore stdout and close log ---
        sys.stdout = original_stdout
        del logger
        print(f"Run {run_id} complete. Logs saved to {log_filepath}")