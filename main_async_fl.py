#  main_async_fl.py (with evaluation)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import copy
import random
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
HP = {
    "data_file": "processed_wisdm.npz",
    "max_updates": 2000,
    "num_clusters": 4,
    "cluster_period": 100,
    "max_concurrent_clients": 10,
    "local_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "test_split_ratio": 0.2, # Hold out 20% of each client's data for testing
}

# --- NEW: Data Loading with Train/Test Split ---
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

# --- Client Update Function ---
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

# --- Helper Functions for Asynchronicity ---
def calculate_alpha(staleness, cluster_size, current_iter):
    time_decay = (0.995)**(current_iter/100); cluster_decay = 0.2/np.log2(cluster_size+3)
    alpha = min(0.2, time_decay * cluster_decay)
    if staleness > 50: alpha /= np.sqrt(staleness-49)
    return max(alpha, 0.01)

def flatten_model_params(model):
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

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

# --- Asynchronous Server Class ---
class AsyncServer:
    def __init__(self, all_client_ids, client_train_loaders, initial_model):
        self.all_client_ids = all_client_ids
        self.client_train_loaders = client_train_loaders
        self.updates_processed = 0
        self.cluster_models = {0: copy.deepcopy(initial_model)}
        self.client_to_cluster = {cid: 0 for cid in all_client_ids}
        self.update_buffer = []
        self.active_clients = {}
        self.free_clients = list(all_client_ids)
    def dispatch_clients(self):
        num_to_dispatch = min(len(self.free_clients), HP["max_concurrent_clients"] - len(self.active_clients))
        for _ in range(num_to_dispatch):
            if not self.free_clients: break
            client_id = self.free_clients.pop(random.randrange(len(self.free_clients)))
            training_time = random.randint(10, 50)
            self.active_clients[client_id] = {"start_iter": self.updates_processed, "finish_iter": self.updates_processed + training_time}

    def process_finished_clients(self):
        finished_ids = [cid for cid, times in self.active_clients.items() if self.updates_processed >= times["finish_iter"]]
        for client_id in finished_ids:
            staleness = self.updates_processed - self.active_clients[client_id]['start_iter']
            
            cluster_id = self.client_to_cluster.get(client_id, 0)
            model_to_train = self.cluster_models.get(cluster_id, list(self.cluster_models.values())[0])

            local_model = client_update(self.client_train_loaders[client_id], model_to_train)
            
            cluster_size = sum(1 for c in self.client_to_cluster.values() if c == cluster_id)
            alpha = calculate_alpha(staleness, cluster_size, self.updates_processed)
            
            with torch.no_grad():
                for cluster_param, local_param in zip(model_to_train.parameters(), local_model.parameters()):
                    cluster_param.data = (1 - alpha) * cluster_param.data + alpha * local_param.data
            
            self.update_buffer.append({"id": client_id, "update_vec": flatten_model_params(local_model)})
            if len(self.update_buffer) > 200: self.update_buffer.pop(0)

            del self.active_clients[client_id]; self.free_clients.append(client_id)
            self.updates_processed += 1
        return len(finished_ids) > 0

    def run_dynamic_clustering(self):
        print(f"\n--- Iteration {self.updates_processed}: Running Clustering ---")
        if len(self.update_buffer) < HP["num_clusters"]: return

        buffer_map = {item['id']: item['update_vec'] for item in self.update_buffer}
        client_ids_in_buffer = list(buffer_map.keys())
        update_vectors = list(buffer_map.values())
        
        similarity_matrix = cosine_similarity(update_vectors)
        
        clustering = AgglomerativeClustering(n_clusters=HP["num_clusters"], linkage='ward')
        cluster_labels = clustering.fit_predict(1 - similarity_matrix)
        
        for i, client_id in enumerate(client_ids_in_buffer):
            self.client_to_cluster[client_id] = cluster_labels[i]
        
        for i in range(HP["num_clusters"]):
            if i not in self.cluster_models:
                self.cluster_models[i] = copy.deepcopy(list(self.cluster_models.values())[0])
        print("Clustering complete.")

    def train(self):
        last_cluster_iter = -1
        pbar = tqdm(total=HP["max_updates"], desc="Processing Updates")
        while self.updates_processed < HP["max_updates"]:
            self.dispatch_clients()
            updates_were_processed = self.process_finished_clients()
            
            current_cluster_period = self.updates_processed // HP["cluster_period"]
            if current_cluster_period > last_cluster_iter:
                self.run_dynamic_clustering(); last_cluster_iter = current_cluster_period
            
            if updates_were_processed:
                pbar.n = min(self.updates_processed, HP["max_updates"]); pbar.refresh()
            elif len(self.active_clients) > 0:
                 self.updates_processed +=1
            elif len(self.free_clients) == len(self.all_client_ids):
                 break
        pbar.close()

# --- Main Execution Block ---
if __name__ == "__main__":
    client_train_loaders, client_test_loaders, all_client_ids = load_data()
    initial_model = HARModel()
    server = AsyncServer(all_client_ids, client_train_loaders, initial_model)
    
    print("\n--- Starting Asynchronous Federated Training (CASA Logic) ---")
    server.train()
    print("\n--- Asynchronous Training Finished ---")
    
    # --- FINAL EVALUATION ---
    print("\n--- Evaluating Final Personalized Models ---")
    client_accuracies = []
    for client_id in all_client_ids:
        if client_id in client_test_loaders:
            cluster_id = server.client_to_cluster[client_id]
            model = server.cluster_models[cluster_id]
            acc = evaluate(model, client_test_loaders[client_id])
            client_accuracies.append(acc)
            print(f"  Client {client_id} (Cluster {cluster_id}): Accuracy = {acc:.4f}")
            
    avg_accuracy = np.mean(client_accuracies)
    std_accuracy = np.std(client_accuracies)
    print("\n--- Evaluation Summary ---")
    print(f"Average Client Accuracy: {avg_accuracy:.4f}")
    print(f"Standard Deviation of Accuracy (Fairness): {std_accuracy:.4f}")