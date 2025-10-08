# preprocess_wisdm.py

import os
import io
import pandas as pd
import numpy as np
# CHANGE: Removed 'from scipy import stats' as it's no longer needed
from tqdm import tqdm

# --- Configuration ---
TIME_STEPS = 200
STEP = 200
DATA_ROOT = "WISDM_ar_latest/WISDM_ar_v1.1"
RAW_DATA_FILE = "WISDM_ar_v1.1_raw.txt"

def load_raw_data(file_path):
    """
    Loads the raw WISDM dataset, counts bad lines, and then processes the clean data.
    """
    print(f"Loading and cleaning data from {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    total_lines = len(lines)
    good_lines = [line for line in lines if line.count(',') == 5]
    bad_lines_count = total_lines - len(good_lines)
    
    print(f"Found and skipped {bad_lines_count} bad/corrupted lines.")

    clean_data_string = "".join(good_lines)
    data_io = io.StringIO(clean_data_string)

    df = pd.read_csv(
        data_io,
        header=None,
        names=['user_id', 'activity', 'timestamp', 'x_accel', 'y_accel', 'z_accel']
    )

    df['z_accel'] = df['z_accel'].astype(str).str.replace(';', '', regex=False).astype(float)
    df.dropna(axis=0, how='any', inplace=True)
    
    df['x_accel'] = pd.to_numeric(df['x_accel'], errors='coerce')
    df['y_accel'] = pd.to_numeric(df['y_accel'], errors='coerce')
    df.dropna(axis=0, how='any', inplace=True)

    print(f"Loaded {len(df)} clean rows of data for processing.")
    return df

def create_windows(df, time_steps, step):
    """
    Creates sliding windows from the raw data DataFrame with a progress bar.
    """
    segments = []
    labels = []
    user_ids = []

    for user, group in tqdm(df.groupby('user_id'), desc="Processing users"):
        activities = group['activity'].unique()
        for activity in activities:
            activity_data = group[group['activity'] == activity]
            for i in range(0, len(activity_data) - time_steps, step):
                x = activity_data['x_accel'].values[i: i + time_steps]
                y = activity_data['y_accel'].values[i: i + time_steps]
                z = activity_data['z_accel'].values[i: i + time_steps]
                
                # CHANGE: Switched from scipy.stats.mode to pandas' built-in .mode() method
                # This correctly handles string labels. We take the first element [0] in case of a tie.
                label = activity_data['activity'][i: i + time_steps].mode()[0]

                segments.append([x, y, z])
                labels.append(label)
                user_ids.append(user)

    reshaped_segments = np.asarray(segments, dtype=np.float32).transpose(0, 2, 1)
    labels_array = np.asarray(labels)
    user_ids_array = np.asarray(user_ids)

    return reshaped_segments, labels_array, user_ids_array

if __name__ == "__main__":
    file_path = os.path.join(DATA_ROOT, RAW_DATA_FILE)
    raw_df = load_raw_data(file_path)

    if raw_df is not None:
        X, y_str, users = create_windows(raw_df, TIME_STEPS, STEP)
        
        activity_map = {label: i for i, label in enumerate(np.unique(y_str))}
        y = np.vectorize(activity_map.get)(y_str)
        
        print("\n--- Preprocessing Complete ---")
        print(f"Shape of data (X): {X.shape}")
        print(f"Shape of labels (y): {y.shape}")
        print(f"Shape of user IDs: {users.shape}")
        print("\nActivity to Integer Mapping:")
        print(activity_map)
        
        client_data = {}
        for user_id in np.unique(users):
            user_mask = (users == user_id)
            client_data[user_id] = {'X': X[user_mask], 'y': y[user_mask]}
        
        print("\n--- Federated Data Partition Created ---")
        print(f"Total number of clients: {len(client_data)}")
        print("Number of samples for first 5 clients:")
        for i, user_id in enumerate(list(client_data.keys())[:5]):
            print(f"  Client {user_id}: {len(client_data[user_id]['y'])} samples")
        
        print("\nSaving processed data to 'processed_wisdm.npz'...")
        np.savez(
            'processed_wisdm.npz',
            X=X,
            y=y,
            users=users,
            activity_map=activity_map
        )
        print("Data saved successfully!")