import json
import numpy as np
import glob
import os

# Find the latest results file automatically
list_of_files = glob.glob('results_reproducibility/all_results_showdown_*.json')
latest_file = max(list_of_files, key=os.path.getctime)

print(f"Analyzing file: {latest_file}")
print("-" * 80)

with open(latest_file, 'r') as f:
    data = json.load(f)

# Organize data by method
methods = {}
for run in data['experiments']:
    name = run['method']
    if name not in methods:
        methods[name] = {'acc': [], 'sparsity': []}
    
    methods[name]['acc'].append(run['avg_accuracy'])
    methods[name]['sparsity'].append(run['avg_sparsity'])

# Print Table
print(f"{'METHOD':<20} | {'AVG ACCURACY (Std)':<25} | {'SPARSITY':<10} | {'SAMPLES':<5}")
print("-" * 80)

for name, metrics in methods.items():
    avg_acc = np.mean(metrics['acc'])
    std_acc = np.std(metrics['acc'])
    avg_spar = np.mean(metrics['sparsity'])
    count = len(metrics['acc'])
    
    print(f"{name:<20} | {avg_acc:.4f} ({std_acc:.4f})      | {avg_spar:.2%}     | {count:<5}")
print("-" * 80)
