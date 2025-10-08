# train_fedavg.py (with evaluation)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import copy
from tqdm import tqdm

# --- Configuration ---
HP = {
    "data_file": "processed_wisdm.npz",
    "num_rounds": 50,
    "clients_per_round": 10,
    "local_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "test_split_ratio": 0.2,
}

# --- Data Loading with Train/Test Split ---
def load_data():
    print("Loading and partitioning data with train/test split...")
    with np.load(HP["data_file"], allow_pickle=True) as data:
        X, y, users = data['X'], data['y'], data['users']
    
    client_train_loaders, client_test_loaders = {}, {}
    all_client_ids = list(np.unique(users))

    # Hold out a global test set from a few users for a single metric
    # More robustly, we test each client on its own held-out data
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
    return local_model.state_dict()

def server_aggregate(global_model_state, local_models_states, client_data_sizes):
    total_samples = sum(client_data_sizes)
    new_global_state = copy.deepcopy(global_model_state)
    for var in new_global_state: new_global_state[var] = torch.zeros_like(new_global_state[var])
    for i, local_state in enumerate(local_models_states):
        weight = client_data_sizes[i] / total_samples
        for var in new_global_state: new_global_state[var] += local_state[var] * weight
    return new_global_state

# --- Evaluation Function ---
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
    
    print("\n--- Starting Federated Training (FedAvg) ---")
    for round_num in range(HP["num_rounds"]):
        selected_client_ids = np.random.choice(all_client_ids, HP["clients_per_round"], replace=False)
        local_models_states, client_data_sizes = [], []
        
        for client_id in tqdm(selected_client_ids, desc=f"Round {round_num+1}/{HP['num_rounds']}"):
            local_state = client_update(client_train_loaders[client_id], global_model)
            local_models_states.append(local_state)
            client_data_sizes.append(len(client_train_loaders[client_id].dataset))
            
        global_model_state = server_aggregate(global_model.state_dict(), local_models_states, client_data_sizes)
        global_model.load_state_dict(global_model_state)
        print(f"Round {round_num+1} complete.\n")

    print("--- Federated Training Finished ---")
    
    # --- FINAL EVALUATION ---
    print("\n--- Evaluating Final Global Model ---")
    client_accuracies = []
    for client_id in all_client_ids:
        if client_id in client_test_loaders:
            # Evaluate the single global model on each client's test data
            acc = evaluate(global_model, client_test_loaders[client_id])
            client_accuracies.append(acc)
            print(f"  Client {client_id}: Accuracy = {acc:.4f}")
            
    avg_accuracy = np.mean(client_accuracies)
    std_accuracy = np.std(client_accuracies)
    print("\n--- Evaluation Summary ---")
    print(f"Average Client Accuracy: {avg_accuracy:.4f}")
    print(f"Standard Deviation of Accuracy (Fairness): {std_accuracy:.4f}")