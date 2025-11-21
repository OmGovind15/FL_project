"""
Results Aggregation & Statistical Analysis for UCI-HAR Dataset
Aggregates results from UCI-HAR experiments
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from glob import glob

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def load_all_results(results_dir='results_uci'):
    """Load all UCI experiment results from directory"""
    results = []
    
    # Find all pickle files
    pkl_files = glob(os.path.join(results_dir, '*.pkl'))
    
    for pkl_file in pkl_files:
        if 'all_results' in pkl_file:
            continue  # Skip aggregated files
        
        try:
            with open(pkl_file, 'rb') as f:
                result = pickle.load(f)
                results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {pkl_file}: {e}")
    
    print(f"Loaded {len(results)} experiment results from UCI-HAR")
    return results


def aggregate_by_method(results):
    """Aggregate results by method across seeds"""
    
    methods = {}
    
    for result in results:
        method = result['method']
        if method not in methods:
            methods[method] = {
                'seeds': [],
                'avg_accuracies': [],
                'std_devs': [],
                'min_accuracies': [],
                'max_accuracies': [],
                'avg_sparsities': [],
                'cluster_0_accs': [],
                'cluster_1_accs': [],
                'cluster_2_accs': [],
            }
        
        methods[method]['seeds'].append(result['seed'])
        methods[method]['avg_accuracies'].append(result['avg_accuracy'])
        methods[method]['std_devs'].append(result['std_dev'])
        methods[method]['min_accuracies'].append(result['min_accuracy'])
        methods[method]['max_accuracies'].append(result['max_accuracy'])
        methods[method]['avg_sparsities'].append(result['avg_sparsity'])
        methods[method]['cluster_0_accs'].append(result['cluster_0_acc'])
        methods[method]['cluster_1_accs'].append(result['cluster_1_acc'])
        methods[method]['cluster_2_accs'].append(result['cluster_2_acc'])
    
    # Calculate statistics
    aggregated = []
    
    for method, data in methods.items():
        if not data['avg_accuracies']:
            continue
        
        # For single seed, std will be 0
        avg_acc_mean = np.mean(data['avg_accuracies'])
        avg_acc_std = np.std(data['avg_accuracies']) if len(data['avg_accuracies']) > 1 else 0.0
        
        aggregated.append({
            'Method': method.upper().replace('_UCI', ''),
            'Avg Acc (%)': f"{avg_acc_mean*100:.2f} ± {avg_acc_std*100:.2f}",
            'Avg Acc Mean': avg_acc_mean,
            'Avg Acc Std': avg_acc_std,
            'Std Dev (%)': f"{np.mean(data['std_devs'])*100:.2f} ± {np.std(data['std_devs'])*100:.2f}",
            'Std Dev Mean': np.mean(data['std_devs']),
            'Min Acc (%)': f"{np.mean(data['min_accuracies'])*100:.2f}",
            'Max Acc (%)': f"{np.mean(data['max_accuracies'])*100:.2f}",
            'Sparsity (%)': f"{np.mean(data['avg_sparsities'])*100:.2f} ± {np.std(data['avg_sparsities'])*100:.2f}",
            'Sparsity Mean': np.mean(data['avg_sparsities']),
            'Cluster 0 (%)': f"{np.mean(data['cluster_0_accs'])*100:.2f}" if method in ['caafp_uci', 'fedchar_uci', 'flcap_uci'] and data['cluster_0_accs'][0] != 0 else 'N/A',
            'Cluster 1 (%)': f"{np.mean(data['cluster_1_accs'])*100:.2f}" if method in ['caafp_uci', 'fedchar_uci', 'flcap_uci'] and data['cluster_1_accs'][0] != 0 else 'N/A',
            'Cluster 2 (%)': f"{np.mean(data['cluster_2_accs'])*100:.2f}" if method in ['caafp_uci', 'fedchar_uci', 'flcap_uci'] and data['cluster_2_accs'][0] != 0 else 'N/A',
            'N Seeds': len(data['seeds']),
            '_raw_accs': data['avg_accuracies'],  # For statistical tests
        })
    
    return pd.DataFrame(aggregated), methods


def perform_statistical_tests(methods_data):
    """Perform pairwise statistical significance tests"""
    
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*70)
    
    method_names = list(methods_data.keys())
    
    # Get CA-AFP results
    if 'caafp_uci' not in methods_data:
        print("Warning: CA-AFP UCI results not found")
        return
    
    caafp_accs = np.array(methods_data['caafp_uci']['avg_accuracies'])
    
    if len(caafp_accs) < 2:
        print("Note: Only 1 seed available. Statistical tests require multiple seeds.")
        print("Showing raw accuracy comparisons instead:\n")
        
        print(f"CA-AFP UCI: {caafp_accs[0]:.4f}")
        for method in method_names:
            if method == 'caafp_uci':
                continue
            baseline_accs = np.array(methods_data[method]['avg_accuracies'])
            if len(baseline_accs) > 0:
                diff = caafp_accs[0] - baseline_accs[0]
                print(f"{method.upper():15} : {baseline_accs[0]:.4f} (diff: {diff:+.4f})")
        return
    
    print(f"\nComparing CA-AFP against all baselines:")
    print(f"CA-AFP accuracies: {caafp_accs}")
    print()
    
    for method in method_names:
        if method == 'caafp_uci':
            continue
        
        baseline_accs = np.array(methods_data[method]['avg_accuracies'])
        
        # Ensure equal length (use minimum length)
        min_len = min(len(caafp_accs), len(baseline_accs))
        caafp_subset = caafp_accs[:min_len]
        baseline_subset = baseline_accs[:min_len]
        
        if min_len < 2:
            print(f"{method.upper():15} : Insufficient data for statistical test")
            continue
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(caafp_subset, baseline_subset)
        
        # Effect size (Cohen's d)
        diff = caafp_subset - baseline_subset
        cohens_d = np.mean(diff) / np.std(diff)
        
        # Interpretation
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"
        
        print(f"{method.upper():15} : t={t_stat:6.3f}, p={p_value:.4f} {sig:3}, Cohen's d={cohens_d:.3f}")
        print(f"                Mean diff: {np.mean(diff)*100:+.2f}%")
    
    if len(caafp_accs) >= 2:
        print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")


def create_visualizations(df, methods_data, w_acc, w_fair, w_sparse, output_dir='results_uci'):
    """Create publication-quality visualizations for UCI-HAR"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Average Accuracy Comparison with Error Bars
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = df['Method'].tolist()
    means = df['Avg Acc Mean'].tolist()
    stds = df['Avg Acc Std'].tolist()
    
    colors = ['#e74c3c' if m == 'CAAFP' else '#3498db' for m in methods]
    
    bars = ax.bar(methods, [m*100 for m in means], yerr=[s*100 for s in stds], 
                   capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Average Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Method', fontsize=13, fontweight='bold')
    ax.set_title('UCI-HAR: Average Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        label = f'{mean*100:.2f}' if std == 0 else f'{mean*100:.2f}±{std*100:.2f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uci_comparison_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/uci_comparison_accuracy.png")
    
    
    # 2. Accuracy vs Sparsity Trade-off
    fig, ax = plt.subplots(figsize=(10, 6))
    method_names = df['Method'].unique()
    colors_map = plt.cm.get_cmap('tab10', len(method_names))
    color_map = {method: colors_map(i) for i, method in enumerate(method_names)}
    
    for _, row in df.iterrows():
        marker = '*' if row['Method'] == 'CAAFP' else 'o'
        size = 300 if row['Method'] == 'CAAFP' else 150
        color = color_map[row['Method']]
        
        ax.scatter(row['Sparsity Mean'], row['Avg Acc Mean']*100, 
                  s=size, marker=marker, color=color, label=row['Method'],
                  alpha=0.8, edgecolors='black', linewidth=1.5)
        
        # Error bars (if multiple seeds)
        if row['Avg Acc Std'] > 0:
            ax.errorbar(row['Sparsity Mean'], row['Avg Acc Mean']*100,
                       yerr=row['Avg Acc Std']*100,
                       fmt='none', color=color, alpha=0.3, capsize=3)
    
    ax.set_xlabel('Model Sparsity (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('UCI-HAR: Accuracy vs Sparsity Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uci_tradeoff_accuracy_sparsity.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/uci_tradeoff_accuracy_sparsity.png")
    
    
    # 3. Per-Cluster Performance (for clustered methods)
    clustered_methods = df[df['Cluster 0 (%)'] != 'N/A'].copy()
    
    if not clustered_methods.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(clustered_methods))
        width = 0.25
        
        # Extract cluster accuracies
        cluster_0 = [float(v.split()[0]) for v in clustered_methods['Cluster 0 (%)']]
        cluster_1 = [float(v.split()[0]) for v in clustered_methods['Cluster 1 (%)']]
        cluster_2 = [float(v.split()[0]) for v in clustered_methods['Cluster 2 (%)']]
        
        bars1 = ax.bar(x - width, cluster_0, width, label='Cluster 0', 
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x, cluster_1, width, label='Cluster 1', 
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
        bars3 = ax.bar(x + width, cluster_2, width, label='Cluster 2', 
                      color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Method', fontsize=13, fontweight='bold')
        ax.set_title('UCI-HAR: Per-Cluster Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(clustered_methods['Method'])
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/uci_comparison_per_cluster.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/uci_comparison_per_cluster.png")
    
    
    # 4. Fairness Comparison (Standard Deviation)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = df['Method'].tolist()
    fairness_means = df['Std Dev Mean'].tolist()
    
    colors = ['#2ecc71' if m == 'CAAFP' else '#e67e22' for m in methods]
    
    bars = ax.bar(methods, [f*100 for f in fairness_means], color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Standard Deviation (%) - Lower is Better', fontsize=13, fontweight='bold')
    ax.set_xlabel('Method', fontsize=13, fontweight='bold')
    ax.set_title('UCI-HAR: Fairness Comparison (Client Performance Variance)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, fairness_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val*100:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uci_comparison_fairness.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/uci_comparison_fairness.png")
    
    
    # 5. Overall "Goodness Score"
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by score for plotting
    df_sorted = df.sort_values('Goodness_Score', ascending=False)
    
    methods = df_sorted['Method'].tolist()
    scores = df_sorted['Goodness_Score'].tolist()
    
    colors = ['#1abc9c' if m == 'CAAFP' else '#95a5a6' for m in methods]
    
    bars = ax.bar(methods, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Overall "Goodness Score" - Higher is Better', fontsize=13, fontweight='bold')
    ax.set_xlabel('Method', fontsize=13, fontweight='bold')
    ax.set_title(f'UCI-HAR: Overall Performance ({w_acc*100:.0f}% Acc, {w_fair*100:.0f}% Fairness, {w_sparse*100:.0f}% Sparsity)', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uci_comparison_goodness_score.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/uci_comparison_goodness_score.png")
    
    plt.close('all')


def generate_latex_table(df, output_dir='results_uci'):
    """Generate LaTeX table for paper"""
    
    # Select columns for paper
    paper_columns = ['Method', 'Avg Acc (%)', 'Std Dev (%)', 
                     'Sparsity (%)', 'Cluster 0 (%)', 'Cluster 1 (%)', 'Cluster 2 (%)']
    
    df_paper = df[paper_columns].copy()
    
    # Generate LaTeX
    latex = df_paper.to_latex(index=False, escape=False)
    
    # Save to file
    with open(f'{output_dir}/uci_results_table.tex', 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to: {output_dir}/uci_results_table.tex")
    print("\nYou can copy this into your paper:")
    print("="*70)
    print(latex)
    print("="*70)


def main():
    """Main aggregation and analysis pipeline for UCI-HAR"""
    
    print("="*70)
    print("UCI-HAR RESULTS AGGREGATION & ANALYSIS")
    print("="*70)
    
    # Load all results
    results = load_all_results()
    
    if not results:
        print("Error: No results found in results_uci/")
        return
    
    # Aggregate by method
    df, methods_data = aggregate_by_method(results)

    # Calculate Goodness Score
    print("\n" + "="*70)
    print("DEFINE 'GOODNESS SCORE' WEIGHTS")
    print("The weights must sum to 1.0.")
    print("="*70)

    while True:
        try:
            w_acc_in = input(f"  Enter weight for Accuracy (0.0 - 1.0): ")
            w_fair_in = input(f"  Enter weight for Fairness (0.0 - 1.0): ")
            w_sparse_in = input(f"  Enter weight for Sparsity (0.0 - 1.0): ")
            
            w_acc = float(w_acc_in)
            w_fair = float(w_fair_in)
            w_sparse = float(w_sparse_in)
            
            total = w_acc + w_fair + w_sparse
            
            if not np.isclose(total, 1.0):
                print(f"\nError: Your weights sum to {total:.2f}, but they must sum to 1.0. Please try again.\n")
                continue
            
            print(f"\nCalculating score with: {w_acc*100:.0f}% Acc, {w_fair*100:.0f}% Fairness, {w_sparse*100:.0f}% Sparsity...")
            break
            
        except ValueError:
            print("\nError: Invalid input. Please enter numbers (e.g., 0.5).\n")
    
    df['Goodness_Score'] = (w_acc * df['Avg Acc Mean']) + \
                           (w_fair * (1 - df['Std Dev Mean'])) + \
                           (w_sparse * df['Sparsity Mean'])
    
    # Sort by the Goodness Score
    df = df.sort_values('Goodness_Score', ascending=False)
    
    # Display results
    print("\n" + "="*70)
    print("UCI-HAR AGGREGATED RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    
    # Statistical tests
    perform_statistical_tests(methods_data)
    
    # Create visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    create_visualizations(df, methods_data, w_acc, w_fair, w_sparse)
    
    # Generate LaTeX table
    generate_latex_table(df)
    
    # Save aggregated results
    df.to_csv('results_uci/uci_aggregated_results.csv', index=False)
    print(f"\nAggregated results saved to: results_uci/uci_aggregated_results.csv")
    
    print("\n" + "="*70)
    print("UCI-HAR ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - uci_aggregated_results.csv")
    print("  - uci_results_table.tex")
    print("  - uci_comparison_accuracy.png")
    print("  - uci_tradeoff_accuracy_sparsity.png")
    print("  - uci_comparison_per_cluster.png (if applicable)")
    print("  - uci_comparison_fairness.png")
    print("  - uci_comparison_goodness_score.png")


if __name__ == "__main__":
    main()
