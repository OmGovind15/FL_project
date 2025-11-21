print("--- LOADING LATEST VERSION OF models.py ---")
import tensorflow as tf
import numpy as np 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow_model_optimization as tfmot

def create_lstm_model(input_shape=(200, 3), num_classes=6):
    """
    Creates a standard LSTM model for HAR.
    
    Args:
        input_shape: tuple, shape of input (timesteps, features)
        num_classes: int, number of activity classes
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(64, return_sequences=True, name='lstm_1', input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, name='lstm_2'),
        Dropout(0.3),
        Dense(32, activation='relu', name='dense_1'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax', name='output')
    ], name='LSTM_HAR_Model')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_prunable_lstm_model(input_shape=(200, 3), num_classes=6, 
                                initial_sparsity=0.0, final_sparsity=0.8,
                                begin_step=0, end_step=1000):
    """
    Creates an LSTM model for manual pruning (without TFMOT wrappers).
    We'll handle pruning manually in the FedMef code.
    """
    # Just create a regular model - we'll handle pruning manually
    model = Sequential([
        LSTM(64, return_sequences=True, name='lstm_1', input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, name='lstm_2'),
        Dropout(0.3),
        Dense(32, activation='relu', name='dense_1'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax', name='output')
    ], name='LSTM_HAR_Model')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def strip_pruning_wrapper(pruned_model):
    """
    Remove pruning wrappers from a pruned model to get the final model.
    
    Args:
        pruned_model: Keras model with pruning wrappers
    
    Returns:
        Keras model without pruning wrappers
    """
    return tfmot.sparsity.keras.strip_pruning(pruned_model)


def get_model_sparsity(model):
    """
    Calculate the sparsity of a model.
    Handles both regular layers and pruning wrapper layers.
    
    Args:
        model: Keras model
    
    Returns:
        float: sparsity percentage
    """
    total_params = 0
    zero_params = 0
    
    for layer in model.layers:
        # Handle pruning wrapper layers
        if hasattr(layer, 'layer'):
            inner_layer = layer.layer
            if hasattr(inner_layer, 'kernel'):
                weights = inner_layer.kernel.numpy()
                total_params += weights.size
                zero_params += (weights == 0).sum()
        # Handle regular layers
        elif hasattr(layer, 'kernel'):
            weights = layer.kernel.numpy()
            total_params += weights.size
            zero_params += (weights == 0).sum()
    
    if total_params == 0:
        return 0.0
    
    return (zero_params / total_params)


def apply_custom_mask(model, mask_dict):
    """
    Apply a custom binary mask to model weights.
    
    Args:
        model: Keras model
        mask_dict: dict mapping layer names to binary masks
    """
    for layer in model.layers:
        # Handle pruning wrapper layers
        if hasattr(layer, 'layer'):
            inner_layer = layer.layer
            if inner_layer.name in mask_dict and hasattr(inner_layer, 'kernel'):
                kernel = inner_layer.kernel.numpy()
                masked_kernel = kernel * mask_dict[inner_layer.name]
                inner_layer.kernel.assign(masked_kernel)
        # Handle regular layers
        elif layer.name in mask_dict and hasattr(layer, 'kernel'):
            kernel = layer.kernel.numpy()
            masked_kernel = kernel * mask_dict[layer.name]
            layer.kernel.assign(masked_kernel)


def apply_pruning_mask(model, sparsity=0.8):
    """
    Manually apply pruning to a model by zeroing out low-magnitude weights.
    
    Args:
        model: Keras model
        sparsity: Target sparsity (0-1)
    
    Returns:
        dict: masks applied to each layer
    """
    masks = {}
    
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            weights = layer.kernel.numpy()
            flat_weights = weights.flatten()
            
            # Determine number of weights to prune
            n_weights = len(flat_weights)
            n_prune = int(n_weights * sparsity)
            
            # Create mask based on magnitude
            threshold_idx = np.argsort(np.abs(flat_weights))[n_prune]
            threshold = np.abs(flat_weights[threshold_idx])
            
            mask = (np.abs(weights) >= threshold).astype(np.float32)
            masks[layer.name] = mask
            
            # Apply mask
            masked_weights = weights * mask
            layer.kernel.assign(masked_weights)
    
    return masks


def get_weight_magnitude_scores(model):
    """
    Get magnitude scores for all weights in the model.
    
    Args:
        model: Keras model
    
    Returns:
        dict: mapping layer names to weight magnitude arrays
    """
    magnitude_scores = {}
    
    for layer in model.layers:
        # Handle pruning wrapper layers
        if hasattr(layer, 'layer'):
            inner_layer = layer.layer
            if hasattr(inner_layer, 'kernel'):
                weights = inner_layer.kernel.numpy()
                magnitudes = tf.abs(weights).numpy()
                magnitude_scores[inner_layer.name] = magnitudes
        # Handle regular layers
        elif hasattr(layer, 'kernel'):
            weights = layer.kernel.numpy()
            magnitudes = tf.abs(weights).numpy()
            magnitude_scores[layer.name] = magnitudes
    
    return magnitude_scores


if __name__ == "__main__":
    # Test the models
    print("Testing standard LSTM model...")
    model = create_lstm_model()
    model.summary()
    print(f"\nModel sparsity: {get_model_sparsity(model):.2f}%")
    
    print("\n" + "="*50)
    print("Testing prunable LSTM model...")
    prunable_model = create_prunable_lstm_model(
        final_sparsity=0.8,
        end_step=1000
    )
    prunable_model.summary()
    print(f"\nModel sparsity before training: {get_model_sparsity(prunable_model):.2f}%")
