import numpy as np
import pandas as pd
import os
import tensorflow as tf
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

def load_uci_har_dataset(root_path='./UCI HAR Dataset'):
    """
    Load the UCI HAR dataset.
    Input Shape: (N, 128, 9) -> 9 channels
    """
    SIGNALS = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]

    def _load_signals(subset):
        signals_data = []
        for signal in SIGNALS:
            filename = f'{root_path}/{subset}/Inertial Signals/{signal}_{subset}.txt'
            try:
                with open(filename, 'r') as f:
                    data = [line.strip().split() for line in f.readlines()]
                signals_data.append(np.array(data, dtype=np.float32))
            except FileNotFoundError:
                print(f"Error: Could not find {filename}. Make sure 'UCI HAR Dataset' is unzipped.")
                sys.exit(1)
        
        return np.transpose(np.array(signals_data), (1, 2, 0))

    def _load_y(subset):
        filename = f'{root_path}/{subset}/y_{subset}.txt'
        with open(filename, 'r') as f:
            y = np.array([line.strip() for line in f.readlines()], dtype=np.int32)
        return y - 1

    def _load_subject(subset):
        filename = f'{root_path}/{subset}/subject_{subset}.txt'
        with open(filename, 'r') as f:
            return np.array([line.strip() for line in f.readlines()], dtype=np.int32)

    print("Loading UCI HAR Train data...")
    X_train = _load_signals('train')
    y_train = _load_y('train')
    sub_train = _load_subject('train')

    print("Loading UCI HAR Test data...")
    X_test = _load_signals('test')
    y_test = _load_y('test')
    sub_test = _load_subject('test')

    X_data = np.concatenate([X_train, X_test])
    y_data = np.concatenate([y_train, y_test])
    user_ids = np.concatenate([sub_train, sub_test])

    print(f"UCI HAR Loaded: {X_data.shape}")
    return X_data, y_data, user_ids

# --- Helper Functions Copied from Original data_loader.py ---

def create_non_iid_data_split(X_data, y_data, user_ids, num_clients=30, alpha=0.5, seed=42):
    """Creates a non-IID data split."""
    print(f"\nCreating non-IID split for {num_clients} clients...")
    np.random.seed(seed)
    
    # UCI HAR has 6 activities (0-5), same as WISDM
    cluster_map = {
        0: [0, 1],  # Cluster 1: Walking, Walking_Upstairs
        1: [2, 3],  # Cluster 2: Walking_Downstairs, Sitting
        2: [4, 5]   # Cluster 3: Standing, Laying
    }
    clients_per_cluster = num_clients // 3
    
    client_data = defaultdict(lambda: {'X': [], 'y': [], 'user_ids': []})
    
    np.random.seed(42)
    sample_counts = np.random.lognormal(mean=4.5, sigma=0.8, size=num_clients)
    sample_counts = np.floor(sample_counts).astype(int) + 50
    
    for cluster_id, activities in cluster_map.items():
        cluster_mask = np.isin(y_data, activities)
        cluster_X = X_data[cluster_mask]
        cluster_y = y_data[cluster_mask]
        cluster_users = user_ids[cluster_mask]
        
        if len(cluster_X) == 0: continue
        
        start_client_idx = cluster_id * clients_per_cluster
        end_client_idx = (cluster_id + 1) * clients_per_cluster
        
        for client_idx in range(start_client_idx, end_client_idx):
            total_samples_for_client = min(sample_counts[client_idx], len(cluster_X))
            proportions = np.random.dirichlet(np.array([alpha] * 2))
            
            num_samples_act1 = int(proportions[0] * total_samples_for_client)
            num_samples_act2 = total_samples_for_client - num_samples_act1
            
            indices_act1 = np.where(cluster_y == activities[0])[0]
            indices_act2 = np.where(cluster_y == activities[1])[0]
            
            if len(indices_act1) > 0 and len(indices_act2) > 0:
                num_samples_act1 = min(num_samples_act1, len(indices_act1))
                num_samples_act2 = min(num_samples_act2, len(indices_act2))
                
                if num_samples_act1 == 0: num_samples_act1 = 1
                if num_samples_act2 == 0: num_samples_act2 = 1
                
                chosen_indices_act1 = np.random.choice(indices_act1, num_samples_act1, replace=(num_samples_act1 > len(indices_act1)))
                chosen_indices_act2 = np.random.choice(indices_act2, num_samples_act2, replace=(num_samples_act2 > len(indices_act2)))
                
                client_X = np.concatenate((cluster_X[chosen_indices_act1], cluster_X[chosen_indices_act2]))
                client_y = np.concatenate((cluster_y[chosen_indices_act1], cluster_y[chosen_indices_act2]))
                
                client_data[client_idx]['X'] = client_X
                client_data[client_idx]['y'] = client_y
            else:
                print(f"WARNING: Client {client_idx} skipped")
    
    return client_data

def create_tf_datasets(client_data, batch_size=32, test_split=0.2):
    """Create TensorFlow datasets."""
    train_datasets = {}
    test_datasets = {}
    
    for client_id, data in client_data.items():
        X = data['X']
        y = data['y']
        
        if len(X) == 0: continue

        n_samples = len(X)
        n_test = int(n_samples * test_split)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        train_datasets[client_id] = train_ds
        test_datasets[client_id] = test_ds
    
    return train_datasets, test_datasets
