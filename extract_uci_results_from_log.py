"""
Extract UCI-HAR results from log file and save as pickle files
"""

import re
import pickle
import json
import os
import numpy as np

def parse_uci_log(log_file='showdown_uci.log'):
    """Parse the UCI log file and extract results"""
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()
    
    results = []
    
    # Pattern to find each method's run
    method_pattern = r"--- Running (\w+) \(Seed (\d+)\) ---"
    result_pattern = r"RESULT (\w+) \(Seed (\d+)\): ([\d.]+)"
    
    # Find all method runs and their results
    method_matches = list(re.finditer(method_pattern, log_content))
    result_matches = list(re.finditer(result_pattern, log_content))
    
    print(f"Found {len(method_matches)} method runs")
    print(f"Found {len(result_matches)} results")
    
    # Parse each result
    for result_match in result_matches:
        method = result_match.group(1)
        seed = int(result_match.group(2))
        avg_accuracy = float(result_match.group(3))
        
        print(f"\nProcessing {method} (seed {seed})")
        
        # Find the detailed stats for this method run
        # Look backwards from the RESULT line
        result_pos = result_match.start()
        
        # Search for statistics in the section before this result
        section_start = max(0, result_pos - 10000)  # Look back 10k chars
        section = log_content[section_start:result_pos]
        
        # Extract statistics
        std_dev = 0.0
        min_acc = 0.0
        max_acc = 1.0
        avg_sparsity = 0.0
        cluster_0_acc = 0.0
        cluster_1_acc = 0.0
        cluster_2_acc = 0.0
        
        # Try to find "Overall Statistics" section
        stats_match = re.search(r"Overall Statistics:.*?Average Accuracy: ([\d.]+).*?Std Dev: ([\d.]+).*?Min: ([\d.]+).*?Max: ([\d.]+)", 
                               section, re.DOTALL)
        
        if stats_match:
        # Only fill std/min/max; DO NOT overwrite avg_accuracy
            std_dev = float(stats_match.group(2))
            min_acc = float(stats_match.group(3))
            max_acc = float(stats_match.group(4))
        else:
            print("  No valid stats found, keeping RESULT accuracy.")        
        # Try to find cluster statistics
        cluster_match = re.search(r"Per-Cluster Statistics:.*?Cluster 0:.*?Avg Acc: ([\d.]+).*?Std Dev: ([\d.]+).*?(?:Sparsity: ([\d.]+)%)?.*?Cluster 1:.*?Avg Acc: ([\d.]+).*?Std Dev: ([\d.]+).*?(?:Sparsity: ([\d.]+)%)?.*?Cluster 2:.*?Avg Acc: ([\d.]+).*?Std Dev: ([\d.]+).*?(?:Sparsity: ([\d.]+)%)?",
                                 section, re.DOTALL)
        
        if cluster_match:
            cluster_0_acc = float(cluster_match.group(1))
            cluster_1_acc = float(cluster_match.group(4))
            cluster_2_acc = float(cluster_match.group(7))
            
            # Get sparsity if available
            if cluster_match.group(3):
                avg_sparsity = float(cluster_match.group(3)) / 100.0
            
            print(f"  Found clusters: C0={cluster_0_acc:.4f}, C1={cluster_1_acc:.4f}, C2={cluster_2_acc:.4f}")
        
        # For lazar, extract sparsity
        if method == 'lazar_uci':
            sparsity_match = re.search(r"Final Sparsity: ([\d.]+)%", section)
            if sparsity_match:
                avg_sparsity = float(sparsity_match.group(1)) / 100.0
                print(f"  Found sparsity: {avg_sparsity:.2%}")
        
        # For fedmef, extract sparsity
        if method == 'fedmef_uci':
            sparsity_match = re.search(r"Final Sparsity: ([\d.]+)%", section)
            if sparsity_match:
                avg_sparsity = float(sparsity_match.group(1)) / 100.0
                print(f"  Found sparsity: {avg_sparsity:.2%}")
        
        # Create result dictionary
        result = {
            'method': method,
            'seed': seed,
            'run_id': f"{method}_seed_{seed}",
            'avg_accuracy': avg_accuracy,
            'std_dev': std_dev,
            'min_accuracy': min_acc,
            'max_accuracy': max_acc,
            'avg_sparsity': avg_sparsity,
            'cluster_0_acc': cluster_0_acc,
            'cluster_1_acc': cluster_1_acc,
            'cluster_2_acc': cluster_2_acc,
            'all_accuracies': [avg_accuracy],  # Single run
        }
        
        results.append(result)
        print(f"  ✓ Saved result for {method}")
    
    return results


def save_results(result, filename):
    """Save results to JSON and pickle"""
    os.makedirs('results_uci', exist_ok=True)
    
    # Save as JSON (human-readable)
    with open(filename + '.json', 'w') as f:
        # Convert numpy types to native Python types
        results_json = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                results_json[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                results_json[k] = float(v)
            else:
                results_json[k] = v
        json.dump(results_json, f, indent=2)
    
    # Save as pickle (for numpy arrays)
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(result, f)
    
    print(f"  Saved to {filename}.json and {filename}.pkl")


def main():
    """Main extraction pipeline"""
    
    print("="*70)
    print("EXTRACTING UCI-HAR RESULTS FROM LOG FILE")
    print("="*70)
    
    # Parse log file
    results = parse_uci_log('showdown_uci.log')
    
    print(f"\n{'='*70}")
    print(f"EXTRACTED {len(results)} RESULTS")
    print("="*70)
    
    # Save each result
    for result in results:
        filename = f"results_uci/{result['run_id']}"
        save_results(result, filename)
    
    # Also save all results together
    all_results = {'experiments': results}
    save_results(all_results, 'results_uci/all_results_uci')
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"Results saved to: results_uci/")
    print(f"\nSummary:")
    for result in results:
        print(f"  {result['method']:15} (seed {result['seed']}): {result['avg_accuracy']:.4f}")
    
    print("\nYou can now run: python aggregate_results_uci.py")


if __name__ == "__main__":
    main()
