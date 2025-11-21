"""
Multi-Seed Experiment Runner
Runs all baselines with multiple random seeds for reproducibility
"""
import glob
from glob import glob
import os
import sys
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
import subprocess
import pickle

# Set base random seeds
SEEDS = [42, 123, 456, 789, 1024]

# Methods to run
METHODS = {
    'caafp': 'final_2.py',
    'fedchar': 'main_fedchar.py',
    'fedmef': 'main_fedmef.py',
    'flcap': 'main_flcap.py',
    'lazar': 'main_lazar.py',
    'caafp_hybrid': 'main_caafp_hybrid.py',    # <-- ADD THIS LINE
    'caafp_hessian': 'main_caafp_hessian.py'   # <-- ADD THIS LINE
}


def set_all_seeds(seed):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f"Set all seeds to {seed}")


def run_single_experiment(method, seed, run_id):
    """
    Run a single experiment with specified method and seed
    
    Args:
        method: Method name (caafp, fedchar, etc.)
        seed: Random seed
        run_id: Unique run identifier
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*70}")
    print(f"Running {method.upper()} with seed {seed}")
    print(f"Run ID: {run_id}")
    print(f"{'='*70}\n")
    
    # Import the appropriate module
    if method == 'caafp':
        from final_2 import run_caafp
        set_all_seeds(seed)
        
        server, clients, results = run_caafp(
            num_clients=30,
            num_rounds=100,
            initial_rounds=10,
            clustering_training_rounds=30,
            clients_per_round=10,
            epochs_per_round=3,
            fine_tune_epochs=3
        )
        
        # Extract metrics
        accuracies = [r['accuracy'] for r in results.values()]
        cluster_results = {i: [] for i in range(3)}
        for client_id, result in results.items():
            cluster_id = server.client_clusters[client_id]
            cluster_results[cluster_id].append(result['accuracy'])
        
        from models import get_model_sparsity
        sparsity = np.mean([get_model_sparsity(m) 
                           for m in server.final_pruned_models.values()])
        
        return {
            'method': method,
            'seed': seed,
            'run_id': run_id,
            'avg_accuracy': np.mean(accuracies),
            'std_dev': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'avg_sparsity': sparsity,
            'cluster_0_acc': np.mean(cluster_results[0]) if cluster_results[0] else 0,
            'cluster_1_acc': np.mean(cluster_results[1]) if cluster_results[1] else 0,
            'cluster_2_acc': np.mean(cluster_results[2]) if cluster_results[2] else 0,
            'all_accuracies': accuracies,
        }
    
    elif method == 'fedchar':
        from main_fedchar import run_fedchar
        set_all_seeds(seed)
        
        server, clients, results = run_fedchar(
            num_clients=30,
            num_rounds=50,
            initial_rounds=10,
            clients_per_round=10,
            epochs_per_round=3
        )
        
        accuracies = [r['accuracy'] for r in results.values()]
        cluster_results = {i: [] for i in range(3)}
        for client_id, result in results.items():
            cluster_id = server.client_clusters[client_id]
            cluster_results[cluster_id].append(result['accuracy'])
        
        return {
            'method': method,
            'seed': seed,
            'run_id': run_id,
            'avg_accuracy': np.mean(accuracies),
            'std_dev': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'avg_sparsity': 0.0,
            'cluster_0_acc': np.mean(cluster_results[0]) if cluster_results[0] else 0,
            'cluster_1_acc': np.mean(cluster_results[1]) if cluster_results[1] else 0,
            'cluster_2_acc': np.mean(cluster_results[2]) if cluster_results[2] else 0,
            'all_accuracies': accuracies,
        }
    
    elif method == 'fedmef':
        from main_fedmef import run_fedmef
        set_all_seeds(seed)
        
        server, clients, results = run_fedmef(
            num_clients=30,
            num_rounds=50,
            adjustment_interval=10,
            stop_adjustment_round=30,
            clients_per_round=10,
            epochs_per_round=3,
            target_sparsity=0.5
        )
        
        accuracies = [r['accuracy'] for r in results.values()]
        from models import get_model_sparsity
        sparsity = get_model_sparsity(server.global_model)
        
        return {
            'method': method,
            'seed': seed,
            'run_id': run_id,
            'avg_accuracy': np.mean(accuracies),
            'std_dev': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'avg_sparsity': sparsity,
            'cluster_0_acc': 0,
            'cluster_1_acc': 0,
            'cluster_2_acc': 0,
            'all_accuracies': accuracies,
        }
    
    elif method == 'flcap':
        from main_flcap import run_flcap
        set_all_seeds(seed)
        
        server, clients, results = run_flcap(
            num_clients=30,
            num_rounds=100,
            initial_rounds=10,
            clustering_training_rounds=30,
            clients_per_round=10,
            epochs_per_round=3,
            fine_tune_epochs=3,
            base_sparsity=0.7,
            random_seed=seed
        )
        
        accuracies = [r['accuracy'] for r in results.values()]
        cluster_results = {i: [] for i in range(3)}
        for client_id, result in results.items():
            cluster_id = server.client_clusters[client_id]
            cluster_results[cluster_id].append(result['accuracy'])
        
        from models import get_model_sparsity
        sparsity = np.mean([get_model_sparsity(m) 
                           for m in server.pruned_models.values()])
        
        return {
            'method': method,
            'seed': seed,
            'run_id': run_id,
            'avg_accuracy': np.mean(accuracies),
            'std_dev': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'avg_sparsity': sparsity,
            'cluster_0_acc': np.mean(cluster_results[0]) if cluster_results[0] else 0,
            'cluster_1_acc': np.mean(cluster_results[1]) if cluster_results[1] else 0,
            'cluster_2_acc': np.mean(cluster_results[2]) if cluster_results[2] else 0,
            'all_accuracies': accuracies,
        }
    
    elif method == 'lazar':
        from main_lazar import run_lazarevich_baseline
        set_all_seeds(seed)
        
        results_0_shot, results_ft, pruned_model = run_lazarevich_baseline(
            num_clients=30,
            total_dense_rounds=40,
            clients_per_round=10,
            epochs_per_round=3,
            target_sparsity=0.5,
            pruning_steps=10,
            calibration_epochs=1,
            fine_tune_epochs=3
        )
        
        # Use fine-tuned results for fair comparison
        accuracies = [r['accuracy'] for r in results_ft.values()]
        from models import get_model_sparsity
        sparsity = get_model_sparsity(pruned_model)
        
        return {
            'method': method,
            'seed': seed,
            'run_id': run_id,
            'avg_accuracy': np.mean(accuracies),
            'std_dev': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'avg_sparsity': sparsity,
            'cluster_0_acc': 0,
            'cluster_1_acc': 0,
            'cluster_2_acc': 0,
            'all_accuracies': accuracies,
        }

    elif method == 'caafp_hybrid':
        from main_caafp_hybrid import run_caafp  # Import from new file
        set_all_seeds(seed)
        
        # Call with new pruning params
        server, clients, results = run_caafp(
            num_clients=30,
            num_rounds=100,
            initial_rounds=10,
            clustering_training_rounds=30,
            clients_per_round=10,
            epochs_per_round=3,
            fine_tune_epochs=3,
            pruning_frequency=5, # New param
            prune_rate=0.05      # New param
        )
        
        # (The rest is identical to the 'caafp' block)
        accuracies = [r['accuracy'] for r in results.values()]
        cluster_results = {i: [] for i in range(3)}
        for client_id, result in results.items():
            cluster_id = server.client_clusters[client_id]
            cluster_results[cluster_id].append(result['accuracy'])
        
        from models import get_model_sparsity
        sparsity = np.mean([get_model_sparsity(m) 
                           for m in server.final_pruned_models.values()])
        
        return {
            'method': method,
            'seed': seed,
            'run_id': run_id,
            'avg_accuracy': np.mean(accuracies),
            'std_dev': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'avg_sparsity': sparsity,
            'cluster_0_acc': np.mean(cluster_results[0]) if cluster_results[0] else 0,
            'cluster_1_acc': np.mean(cluster_results[1]) if cluster_results[1] else 0,
            'cluster_2_acc': np.mean(cluster_results[2]) if cluster_results[2] else 0,
            'all_accuracies': accuracies,
        }

    elif method == 'caafp_hessian':
        from main_caafp_hessian import run_caafp  # Import from new file
        set_all_seeds(seed)
        
        # Call with new pruning params
        server, clients, results = run_caafp(
            num_clients=30,
            num_rounds=100,
            initial_rounds=10,
            clustering_training_rounds=30,
            clients_per_round=10,
            epochs_per_round=3,
            fine_tune_epochs=3,
            pruning_frequency=5, # New param
            prune_rate=0.05      # New param
        )
        
        # (The rest is identical to the 'caafp' block)
        accuracies = [r['accuracy'] for r in results.values()]
        cluster_results = {i: [] for i in range(3)}
        for client_id, result in results.items():
            cluster_id = server.client_clusters[client_id]
            cluster_results[cluster_id].append(result['accuracy'])
        
        from models import get_model_sparsity
        sparsity = np.mean([get_model_sparsity(m) 
                           for m in server.final_pruned_models.values()])
        
        return {
            'method': method,
            'seed': seed,
            'run_id': run_id,
            'avg_accuracy': np.mean(accuracies),
            'std_dev': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'avg_sparsity': sparsity,
            'cluster_0_acc': np.mean(cluster_results[0]) if cluster_results[0] else 0,
            'cluster_1_acc': np.mean(cluster_results[1]) if cluster_results[1] else 0,
            'cluster_2_acc': np.mean(cluster_results[2]) if cluster_results[2] else 0,
            'all_accuracies': accuracies,
        }
def save_results(results, filename):
    """Save results to JSON and pickle"""
    # Save as JSON (human-readable)
    with open(filename + '.json', 'w') as f:
        # Convert numpy types to native Python types
        results_json = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                results_json[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                results_json[k] = float(v)
            else:
                results_json[k] = v
        json.dump(results_json, f, indent=2)
    
    # Save as pickle (for numpy arrays)
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filename}.json and {filename}.pkl")


def run_all_experiments():
    """Run all methods with all seeds"""
    
    # Create directories
    os.makedirs('results_reproducibility', exist_ok=True)
    os.makedirs('logs_reproducibility', exist_ok=True)
    
    # Master results storage
    all_results = []
    
    # Run each method with each seed
    for method in METHODS.keys():
        for seed in SEEDS:
            run_id = f"{method}_seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            base_filename = f"results_reproducibility/{method}_seed_{seed}"

            # Use glob to check for any existing file starting with this pattern
            if glob(base_filename + '_*.pkl'):
                print(f"SKIPPING: {method} with seed {seed} already completed.")
                continue
            try:
                # Run experiment
                result = run_single_experiment(method, seed, run_id)
                all_results.append(result)
                
                # Save individual result
                save_results(
                    result, 
                    f'results_reproducibility/{run_id}'
                )
                
                print(f"\n✓ Completed: {method} with seed {seed}")
                print(f"  Avg Accuracy: {result['avg_accuracy']:.4f}")
                print(f"  Std Dev: {result['std_dev']:.4f}")
                print(f"  Sparsity: {result['avg_sparsity']:.2f}%\n")
                
            except Exception as e:
                print(f"\n✗ Failed: {method} with seed {seed}")
                print(f"  Error: {str(e)}\n")
                continue
    
    # Save all results
    save_results(
        {'experiments': all_results}, 
        f'results_reproducibility/all_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    return all_results


if __name__ == "__main__":
    print("="*70)
    print("MULTI-SEED REPRODUCIBILITY EXPERIMENTS")
    print("="*70)
    print(f"Methods: {list(METHODS.keys())}")
    print(f"Seeds: {SEEDS}")
    print(f"Total experiments: {len(METHODS) * len(SEEDS)}")
    print("="*70)
    
    # Run all experiments
    results = run_all_experiments()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Total successful runs: {len(results)}")
    print("Results saved in: results_reproducibility/")
