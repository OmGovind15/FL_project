import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def load_wisdm_dataset(filepath='WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'):
    """
    Load and preprocess the WISDM dataset.
    
    Returns:
        X_data: numpy array of features (samples, timesteps, features)
        y_data: numpy array of labels
        user_ids: numpy array of user IDs
    """
    # Load raw data
    column_names = ['user_id', 'activity', 'timestamp', 'x', 'y', 'z']
    
    # Read the file, handling potential formatting issues
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            # Remove trailing semicolon and whitespace
            line = line.strip().rstrip(';').strip()
            if line:
                values = line.split(',')
                # Clean each value
                values = [v.strip() for v in values]
                # Only keep rows with exactly 6 non-empty values
                if len(values) == 6 and all(v for v in values):
                    data.append(values)
    
    if not data:
        raise ValueError(f"No valid data found in {filepath}")
    
    df = pd.DataFrame(data, columns=column_names)
    
    # Convert data types with error handling
    try:
        df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df['z'] = pd.to_numeric(df['z'], errors='coerce')
        
        # Remove rows with any NaN values
        df = df.dropna()
        
        # Convert to proper types after cleaning
        df['user_id'] = df['user_id'].astype(int)
        df['timestamp'] = df['timestamp'].astype(int)
        df['x'] = df['x'].astype(float)
        df['y'] = df['y'].astype(float)
        df['z'] = df['z'].astype(float)
    except Exception as e:
        print(f"Error converting data types: {e}")
        print(f"Sample of problematic data:")
        print(df.head(10))
        raise
    
    # Map activities to numeric labels
    activity_map = {
        'Walking': 0,
        'Jogging': 1,
        'Upstairs': 2,
        'Downstairs': 3,
        'Sitting': 4,
        'Standing': 5
    }
    
    # Map activities, handle any unknown activities
    df['activity_label'] = df['activity'].map(activity_map)
    df = df.dropna(subset=['activity_label'])  # Remove unknown activities
    df['activity_label'] = df['activity_label'].astype(int)
    
    print(f"Loaded {len(df)} valid samples")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique activities: {df['activity'].unique()}")
    
    # Create sliding windows (200 samples with 50% overlap)
    window_size = 200
    step_size = 100
    
    X_data = []
    y_data = []
    user_ids = []
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('timestamp')
        
        for activity in user_data['activity_label'].unique():
            activity_data = user_data[user_data['activity_label'] == activity]
            
            # Extract accelerometer readings
            accel_data = activity_data[['x', 'y', 'z']].values
            
            # Create windows only if we have enough data
            if len(accel_data) < window_size:
                continue
            
            # Create windows
            for i in range(0, len(accel_data) - window_size + 1, step_size):
                window = accel_data[i:i + window_size]
                if len(window) == window_size:
                    X_data.append(window)
                    y_data.append(activity)
                    user_ids.append(user_id)
    
    if not X_data:
        raise ValueError("No valid windows created. Check if data has enough samples.")
    
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    user_ids = np.array(user_ids)
    
    print(f"Created {len(X_data)} windows")
    
    # Normalize features
    scaler = StandardScaler()
    X_data_reshaped = X_data.reshape(-1, 3)
    X_data_normalized = scaler.fit_transform(X_data_reshaped)
    X_data = X_data_normalized.reshape(-1, window_size, 3)
    
    return X_data, y_data, user_ids


def create_non_iid_data_split(X_data, y_data, user_ids, num_clients=30, alpha=0.5, seed=42):
    """
    Creates a non-IID data split with feature, quantity, and label skew.
    
    Cluster 1 (clients 0-9): Walking, Jogging
    Cluster 2 (clients 10-19): Upstairs, Downstairs
    Cluster 3 (clients 20-29): Sitting, Standing
    """
    print(f"\nCreating non-IID split for {num_clients} clients...")
    np.random.seed(seed)
    # Define the hard cluster split based on activities
    cluster_map = {
        0: [0, 1],  # Cluster 1: Walking, Jogging
        1: [2, 3],  # Cluster 2: Upstairs, Downstairs
        2: [4, 5]   # Cluster 3: Sitting, Standing
    }
    clients_per_cluster = num_clients // 3
    
    client_data = defaultdict(lambda: {'X': [], 'y': [], 'user_ids': []})
    
    # --- Introduce Quantity Skew ---
    # Use a log-normal distribution to get varied sample counts per client
    np.random.seed(42)
    sample_counts = np.random.lognormal(mean=4.5, sigma=0.8, size=num_clients)
    sample_counts = np.floor(sample_counts).astype(int) + 50  # Min 50 samples
    
    for cluster_id, activities in cluster_map.items():
        # Get all data for the activities in this cluster
        cluster_mask = np.isin(y_data, activities)
        cluster_X = X_data[cluster_mask]
        cluster_y = y_data[cluster_mask]
        cluster_users = user_ids[cluster_mask]
        
        print(f"Cluster {cluster_id} (activities {activities}): {len(cluster_X)} total samples")
        
        if len(cluster_X) == 0:
            print(f"WARNING: No data for cluster {cluster_id}!")
            continue
        
        start_client_idx = cluster_id * clients_per_cluster
        end_client_idx = (cluster_id + 1) * clients_per_cluster
        
        for client_idx in range(start_client_idx, end_client_idx):
            total_samples_for_client = min(sample_counts[client_idx], len(cluster_X))
            
            # --- Introduce Label Distribution Skew ---
            # Use Dirichlet to get a skewed proportion for the two activities
            proportions = np.random.dirichlet(np.array([alpha] * 2))
            
            # Allocate samples based on skewed proportions
            num_samples_act1 = int(proportions[0] * total_samples_for_client)
            num_samples_act2 = total_samples_for_client - num_samples_act1
            
            # Get indices for each activity within the cluster's data
            indices_act1 = np.where(cluster_y == activities[0])[0]
            indices_act2 = np.where(cluster_y == activities[1])[0]
            
            # Randomly sample the data (with validation)
            if len(indices_act1) > 0 and len(indices_act2) > 0:
                # Ensure we don't request more samples than available
                num_samples_act1 = min(num_samples_act1, len(indices_act1))
                num_samples_act2 = min(num_samples_act2, len(indices_act2))
                
                # Make sure we have at least 1 sample from each activity
                if num_samples_act1 == 0:
                    num_samples_act1 = 1
                    num_samples_act2 = total_samples_for_client - 1
                if num_samples_act2 == 0:
                    num_samples_act2 = 1
                    num_samples_act1 = total_samples_for_client - 1
                
                chosen_indices_act1 = np.random.choice(
                    indices_act1, 
                    num_samples_act1, 
                    replace=(num_samples_act1 > len(indices_act1))
                )
                chosen_indices_act2 = np.random.choice(
                    indices_act2, 
                    num_samples_act2, 
                    replace=(num_samples_act2 > len(indices_act2))
                )
                
                client_X = np.concatenate((
                    cluster_X[chosen_indices_act1], 
                    cluster_X[chosen_indices_act2]
                ))
                client_y = np.concatenate((
                    cluster_y[chosen_indices_act1], 
                    cluster_y[chosen_indices_act2]
                ))
                client_user_ids = np.concatenate((
                    cluster_users[chosen_indices_act1],
                    cluster_users[chosen_indices_act2]
                ))
                
                client_data[client_idx]['X'] = client_X
                client_data[client_idx]['y'] = client_y
                client_data[client_idx]['user_ids'] = client_user_ids
            else:
                print(f"WARNING: Client {client_idx} skipped - insufficient data for both activities")
    
    print(f"Successfully created data for {len(client_data)} clients")
    return client_data


def create_tf_datasets(client_data, batch_size=32, test_split=0.2):
    """
    Create TensorFlow datasets for each client.
    
    Returns:
        train_datasets: dict of client_id -> tf.data.Dataset (training)
        test_datasets: dict of client_id -> tf.data.Dataset (testing)
    """
    train_datasets = {}
    test_datasets = {}
    
    for client_id, data in client_data.items():
        X = data['X']
        y = data['y']
        
        # Split into train and test
        n_samples = len(X)
        n_test = int(n_samples * test_split)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Create TF datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        train_datasets[client_id] = train_ds
        test_datasets[client_id] = test_ds
    
    return train_datasets, test_datasets


if __name__ == "__main__":
    # Test the data loader
    print("Loading WISDM dataset...")
    X_data, y_data, user_ids = load_wisdm_dataset()
    print(f"Dataset shape: X={X_data.shape}, y={y_data.shape}")
    print(f"Unique activities: {np.unique(y_data)}")
    print(f"Unique users: {len(np.unique(user_ids))}")
    
    print("\nCreating non-IID split...")
    client_data = create_non_iid_data_split(X_data, y_data, user_ids, num_clients=30)
    
    print(f"Number of clients: {len(client_data)}")
    for i in range(0, 30, 10):
        print(f"\nCluster {i//10} (Clients {i}-{i+9}):")
        for client_id in range(i, min(i+10, 30)):
            if client_id in client_data:
                print(f"  Client {client_id}: {len(client_data[client_id]['X'])} samples, "
                      f"activities: {np.unique(client_data[client_id]['y'])}")
