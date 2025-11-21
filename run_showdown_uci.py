import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import pickle

# --- CONFIGURATION ---
# We will run on 3 seeds for a robust generalizability check
SEEDS = [42] 

METHODS = {
    'lazar_uci':   'main_lazar_uci.py',
    
}
# ---------------------

def set_all_seeds(seed):
    import random
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message); self.logfile.write(message); self.flush()
    def flush(self):
        self.terminal.flush(); self.logfile.flush()

if __name__ == "__main__":
    os.makedirs('results_uci', exist_ok=True)
    os.makedirs('logs_uci', exist_ok=True)
    
    print("="*70)
    print(f"STARTING UCI-HAR GENERALIZABILITY TEST (Seeds: {SEEDS})")
    print("="*70)

    # Storage for final summary
    final_summary = {method: [] for method in METHODS}

    for seed in SEEDS:
        print(f"\n\n{'='*40}")
        print(f"   STARTING SEED {seed}")
        print(f"{'='*40}")

        for name, script_file in METHODS.items():
            run_id = f"{name}_seed_{seed}"
            log_path = f"logs_uci/{run_id}.log"
            
            # Reset stdout for logging
            original_stdout = sys.stdout
            sys.stdout = Logger(log_path)
            
            print(f"\n--- Running {name} (Seed {seed}) ---")
            try:
                set_all_seeds(seed)
                
                # --- Dynamic Imports ---
                if name == 'caafp_uci':
                    from main_caafp_uci import run_caafp
                    # Note: Using same params as your winning WISDM run
                    server, _, results = run_caafp(
                        num_clients=30, num_rounds=40, initial_rounds=5,
                        clustering_training_rounds=35, clients_per_round=10,
                        epochs_per_round=3, fine_tune_epochs=3,
                        prune_rate=0.02 # Your winning rate
                    )
                    accs = [r['accuracy'] for r in results.values()]

                elif name == 'flcap_uci':
                    from main_flcap_uci import run_flcap
                    server, _, results = run_flcap(
                        num_clients=30, num_rounds=40, initial_rounds=5,
                        clustering_training_rounds=35, clients_per_round=10,
                        epochs_per_round=3, fine_tune_epochs=3,
                        base_sparsity=0.7, random_seed=seed
                    )
                    accs = [r['accuracy'] for r in results.values()]

                elif name == 'fedchar_uci':
                    from main_fedchar_uci import run_fedchar
                    server, _, results = run_fedchar(
                        num_clients=30, num_rounds=50, initial_rounds=5,
                        clients_per_round=10, epochs_per_round=3
                    )
                    accs = [r['accuracy'] for r in results.values()]

                elif name == 'lazar_uci':
                    from main_lazar_uci import run_lazarevich_baseline
    
                    # Run LAZAR
                    results_0, results_ft, _ = run_lazarevich_baseline(
                    num_clients=30, total_dense_rounds=40,
                    clients_per_round=10, epochs_per_round=3,
                    target_sparsity=0.7, pruning_steps=10,
                    calibration_epochs=1, fine_tune_epochs=3
                    )

    # ------------------------------------------------------------
    # ★ NEW: PRINT PER-CLIENT ACCURACY (like other models)
    # ------------------------------------------------------------
                    print("\n======================")
                    print("LAZAR: Per-client accuracies (Fine-tuned)")
                    print("======================")
    
                    for cid, res in sorted(results_ft.items()):
                        print(f"Client {cid:2d}: Accuracy = {res['accuracy']:.4f}")

    # ------------------------------------------------------------
    # ★ NEW: SAVE PER-CLIENT ACCURACIES TO DISK
    # ------------------------------------------------------------
                    import json
                    save_path = f"results_uci/lazar_seed_{seed}_per_client.json"
                    with open(save_path, "w") as f:
                        json.dump(results_ft, f, indent=4)
    
                    print(f"\nSaved LAZAR per-client results to: {save_path}\n")

    # ------------------------------------------------------------
    # Extract accuracy list for averaging (as before)
    # ------------------------------------------------------------
                    accs = [res['accuracy'] for res in results_ft.values()]
                elif name == 'fedmef_uci':
                    from main_fedmef_uci import run_fedmef
                    server, _, results = run_fedmef(
                        num_clients=30, num_rounds=50, adjustment_interval=10,
                        stop_adjustment_round=30, clients_per_round=10,
                        epochs_per_round=3, target_sparsity=0.7
                    )
                    accs = [r['accuracy'] for r in results.values()]

                # Calculate and store result
                avg_acc = np.mean(accs)
                final_summary[name].append(avg_acc)
                print(f"RESULT {name} (Seed {seed}): {avg_acc:.4f}")

            except Exception as e:
                print(f"FAILED {name}: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                sys.stdout = original_stdout

    # --- Print Final Summary Table ---
    print("\n\n" + "="*70)
    print("UCI-HAR 3-SEED SUMMARY")
    print("="*70)
    print(f"{'METHOD':<15} | {'AVG ACCURACY':<15} | {'STD DEV':<10} | {'RAW SEEDS'}")
    print("-" * 70)
    
    for name, scores in final_summary.items():
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            raw_str = ", ".join([f"{s:.4f}" for s in scores])
            print(f"{name:<15} | {mean_score:.4f}          | {std_score:.4f}     | {raw_str}")
        else:
             print(f"{name:<15} | FAILED")
    print("="*70)
