# main_clustered_fl.py (with evaluation)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import copy
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
HP = {
    "data_file": "processed_wisdm.npz",
    "num_rounds": 50,
    "initial_rounds": 10,       # Rounds of FedAvg before clustering
    "num_clusters": 4,          # Pre-defined number of clusters
    "clients_per_round": 10,
    "local_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "test_split_ratio": 0.2,    # Hold out 20% of each client's data for testing
}

# --- Data Loading with Train/Test Split ---
def load_data():
    print("Loading and partitioning data with train/test split...")
    with np.load(HP["data_file"], allow_pickle=True) as data:
        X, y, users = data['X'], data['y'], data['users']
    
    client_train_loaders, client_test_loaders = {}, {}
    all_client_ids = list(np.unique(users))

    for user_id in all_client_ids:
        user_mask = (users == user_id)
        X_user, y_user = X[user_mask], y[user_mask]
        
        X_tensor = torch.tensor(X_user, dtype=torch.float32).permute(0, 2, 1)
        y_tensor = torch.tensor(y_user, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        
        test_size = int(len(dataset) * HP["test_split_ratio"])
        train_size = len(dataset) - test_size
        
        if train_size > 0 and test_size > 0:
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            client_train_loaders[user_id] = DataLoader(train_dataset, batch_size=HP["batch_size"], shuffle=True)
            client_test_loaders[user_id] = DataLoader(test_dataset, batch_size=HP["batch_size"], shuffle=False)
    
    # Remove clients that didn't have enough data for a train/test split
    valid_clients = list(client_train_loaders.keys())
    print(f"Data loaded for {len(valid_clients)} clients with sufficient data.")
    return client_train_loaders, client_test_loaders, valid_clients

# --- Model Definition ---
class HARModel(nn.Module):
    def __init__(self, num_classes=6):
        super(HARModel, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, 3, 1, 1); self.relu1 = nn.ReLU(); self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(32, 64, 3, 1, 1); self.relu2 = nn.ReLU(); self.pool2 = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten(); self.fc1 = nn.Linear(64*50, 128); self.relu3 = nn.ReLU(); self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x))); x = self.pool2(self.relu2(self.conv2(x)))
        return self.fc2(self.relu3(self.fc1(self.flatten(x))))

# --- Federated Learning Functions ---
def client_update(client_loader, model):
    local_model = copy.deepcopy(model)
    local_model.train()
    optimizer = optim.Adam(local_model.parameters(), lr=HP["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    for _ in range(HP["local_epochs"]):
        for data, target in client_loader:
            optimizer.zero_grad(); output = local_model(data); loss = criterion(output, target)
            loss.backward(); optimizer.step()
    return local_model

def server_aggregate(local_models, client_data_sizes):
    total_samples = sum(client_data_sizes)
    global_model = HARModel()
    global_state = global_model.state_dict()
    for var in global_state: global_state[var] = torch.zeros_like(global_state[var])
    for i, model in enumerate(local_models):
        weight = client_data_sizes[i] / total_samples
        for var in global_state: global_state[var] += model.state_dict()[var] * weight
    global_model.load_state_dict(global_state)
    return global_model

def perform_clustering(all_client_ids, client_loaders, global_model):
    print("\n--- Performing Client Clustering ---")
    model_updates = []
    initial_params = torch.cat([p.view(-1) for p in global_model.state_dict().values()])
    
    for client_id in tqdm(all_client_ids, desc="Getting client updates"):
        local_model = client_update(client_loaders[client_id], global_model)
        local_params = torch.cat([p.view(-1) for p in local_model.state_dict().values()])
        model_updates.append((local_params - initial_params).numpy())
        
    model_updates = np.array(model_updates)
    similarity_matrix = cosine_similarity(model_updates)
    
    clustering = AgglomerativeClustering(n_clusters=HP["num_clusters"], linkage='ward')
    cluster_labels = clustering.fit_predict(1 - similarity_matrix)
    
    client_to_cluster = {client_id: label for client_id, label in zip(all_client_ids, cluster_labels)}
    
    print("Clustering complete.")
    for i in range(HP["num_clusters"]):
        clients_in_cluster = [cid for cid, clabel in client_to_cluster.items() if clabel == i]
        print(f"  Cluster {i}: {len(clients_in_cluster)} clients")
    return client_to_cluster

# --- NEW: Evaluation Function ---
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

# --- Main Training Loop ---
if __name__ == "__main__":
    client_train_loaders, client_test_loaders, all_client_ids = load_data()
    global_model = HARModel()
    
    print("\n--- Starting Initial FedAvg Rounds ---")
    for round_num in range(HP["initial_rounds"]):
        selected_client_ids = np.random.choice(all_client_ids, HP["clients_per_round"], replace=False)
        local_models = [client_update(client_train_loaders[cid], global_model) for cid in selected_client_ids]
        client_sizes = [len(client_train_loaders[cid].dataset) for cid in selected_client_ids]
        global_model = server_aggregate(local_models, client_sizes)
        print(f"Initial Round {round_num+1}/{HP['initial_rounds']} complete.")

    client_to_cluster = perform_clustering(all_client_ids, client_train_loaders, global_model)
    cluster_models = {i: copy.deepcopy(global_model) for i in range(HP["num_clusters"])}
    
    print("\n--- Starting Clustered Training Rounds ---")
    for round_num in range(HP["initial_rounds"], HP["num_rounds"]):
        print(f"Clustered Round {round_num+1}/{HP['num_rounds']}")
        for cluster_id in range(HP["num_clusters"]):
            clients_in_cluster = [cid for cid, clabel in client_to_cluster.items() if clabel == cluster_id]
            if not clients_in_cluster: continue
            
            num_to_select = min(len(clients_in_cluster), max(1, HP["clients_per_round"] // HP["num_clusters"]))
            selected_client_ids = np.random.choice(clients_in_cluster, num_to_select, replace=False)
            
            cluster_model = cluster_models[cluster_id]
            local_models = [client_update(client_train_loaders[cid], cluster_model) for cid in selected_client_ids]
            client_sizes = [len(client_train_loaders[cid].dataset) for cid in selected_client_ids]
            cluster_models[cluster_id] = server_aggregate(local_models, client_sizes)
    
    print("\n--- Clustered Federated Training Finished ---")
    
    # --- FINAL EVALUATION ---
    print("\n--- Evaluating Final Personalized Models ---")
    client_accuracies = []
    for client_id in all_client_ids:
        if client_id in client_test_loaders:
            cluster_id = client_to_cluster[client_id]
            model = cluster_models[cluster_id]
            acc = evaluate(model, client_test_loaders[client_id])
            client_accuracies.append(acc)
            print(f"  Client {client_id} (Cluster {cluster_id}): Accuracy = {acc:.4f}")
            
    avg_accuracy = np.mean(client_accuracies)
    std_accuracy = np.std(client_accuracies)
    print("\n--- Evaluation Summary ---")
    print(f"Average Client Accuracy: {avg_accuracy:.4f}")
    print(f"Standard Deviation of Accuracy (Fairness): {std_accuracy:.4f}")