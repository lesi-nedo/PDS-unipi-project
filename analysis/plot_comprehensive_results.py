#!/usr/bin/env python3
"""
Comprehensive Performance Analysis and Visualization
Generates plots for speedup, efficiency, and scalability analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_comprehensive_results(csv_path):
    """Load comprehensive performance results from CSV"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def plot_speedup_curves(df, output_dir, phase='Training'):
    """Plot speedup curves for different configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{phase} Speedup Analysis', fontsize=16, fontweight='bold')
    
    # Get unique configurations
    tree_counts = sorted(df['NumTrees'].unique())
    samples = sorted(df['SamplesPerTree'].unique())
    
    speedup_col = f'{phase}Speedup'
    
    # Plot 1: Speedup vs Threads for different tree counts
    ax = axes[0, 0]
    for trees in tree_counts:
        subset = df[df['NumTrees'] == trees].groupby('NumThreads')[speedup_col].mean()
        ax.plot(subset.index, subset.values, marker='o', label=f'{trees} trees', linewidth=2)
    
    # Add ideal speedup line
    max_threads = df['NumThreads'].max()
    ax.plot([1, max_threads], [1, max_threads], 'k--', label='Ideal Speedup', linewidth=2)
    
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Speedup', fontweight='bold')
    ax.set_title(f'{phase} Speedup vs Thread Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    # Plot 2: Speedup vs Threads for different sample sizes
    ax = axes[0, 1]
    for sample in samples:
        subset = df[df['SamplesPerTree'] == sample].groupby('NumThreads')[speedup_col].mean()
        ax.plot(subset.index, subset.values, marker='s', label=f'{sample} samples', linewidth=2)
    
    ax.plot([1, max_threads], [1, max_threads], 'k--', label='Ideal Speedup', linewidth=2)
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Speedup', fontweight='bold')
    ax.set_title(f'{phase} Speedup vs Thread Count (by Dataset Size)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    # Plot 3: Heatmap of speedup
    ax = axes[1, 0]
    pivot = df.pivot_table(values=speedup_col, index='NumTrees', columns='NumThreads', aggfunc='mean')
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Number of Trees', fontweight='bold')
    ax.set_title(f'{phase} Speedup Heatmap')
    plt.colorbar(im, ax=ax, label='Speedup')
    
    # Plot 4: Speedup improvement rate
    ax = axes[1, 1]
    for trees in tree_counts[:3]:  # Show only first 3 for clarity
        subset = df[df['NumTrees'] == trees].groupby('NumThreads')[speedup_col].mean()
        if len(subset) > 1:
            improvement = subset.pct_change() * 100
            ax.plot(subset.index[1:], improvement.values[1:], marker='o', label=f'{trees} trees', linewidth=2)
    
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Speedup Improvement (%)', fontweight='bold')
    ax.set_title(f'{phase} Speedup Improvement Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = Path(output_dir) / f'{phase.lower()}_speedup_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_efficiency_curves(df, output_dir, phase='Training'):
    """Plot efficiency curves"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{phase} Efficiency Analysis', fontsize=16, fontweight='bold')
    
    tree_counts = sorted(df['NumTrees'].unique())
    efficiency_col = f'{phase}Efficiency'
    
    # Plot 1: Efficiency vs Threads
    ax = axes[0, 0]
    for trees in tree_counts:
        subset = df[df['NumTrees'] == trees].groupby('NumThreads')[efficiency_col].mean()
        ax.plot(subset.index, subset.values * 100, marker='o', label=f'{trees} trees', linewidth=2)
    
    ax.axhline(y=100, color='k', linestyle='--', label='Ideal (100%)', linewidth=2)
    ax.axhline(y=80, color='g', linestyle=':', label='Good (80%)', linewidth=1.5)
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontweight='bold')
    ax.set_title(f'{phase} Parallel Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # Plot 2: Efficiency drop-off
    ax = axes[0, 1]
    avg_efficiency = df.groupby('NumThreads')[efficiency_col].mean() * 100
    ax.bar(range(len(avg_efficiency)), avg_efficiency.values, 
           color=['green' if e >= 80 else 'orange' if e >= 60 else 'red' for e in avg_efficiency.values])
    ax.set_xticks(range(len(avg_efficiency)))
    ax.set_xticklabels(avg_efficiency.index)
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Average Efficiency (%)', fontweight='bold')
    ax.set_title(f'Average {phase} Efficiency by Thread Count')
    ax.axhline(y=80, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Efficiency vs Problem Size
    ax = axes[1, 0]
    samples = sorted(df['SamplesPerTree'].unique())
    for sample in samples:
        subset = df[df['SamplesPerTree'] == sample].groupby('NumThreads')[efficiency_col].mean()
        ax.plot(subset.index, subset.values * 100, marker='s', label=f'{sample} samples', linewidth=2)
    
    ax.axhline(y=80, color='g', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontweight='bold')
    ax.set_title(f'{phase} Efficiency vs Dataset Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # Plot 4: Efficiency heatmap
    ax = axes[1, 1]
    pivot = df.pivot_table(values=efficiency_col, index='NumTrees', columns='NumThreads', aggfunc='mean')
    im = ax.imshow(pivot.values * 100, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Number of Trees', fontweight='bold')
    ax.set_title(f'{phase} Efficiency Heatmap (%)')
    plt.colorbar(im, ax=ax, label='Efficiency (%)')
    
    plt.tight_layout()
    output_path = Path(output_dir) / f'{phase.lower()}_efficiency_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_scalability_analysis(df, output_dir):
    """Plot strong and weak scalability analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scalability Analysis', fontsize=16, fontweight='bold')
    
    # Strong Scaling - Training
    ax = axes[0, 0]
    tree_counts = sorted(df['NumTrees'].unique())
    for trees in tree_counts:
        subset = df[df['NumTrees'] == trees].groupby('NumThreads')['TrainingTime_ms'].mean()
        baseline = subset.iloc[0]
        strong_scaling = baseline / subset
        ax.plot(subset.index, strong_scaling, marker='o', label=f'{trees} trees', linewidth=2)
    
    max_threads = df['NumThreads'].max()
    ax.plot([1, max_threads], [1, max_threads], 'k--', label='Ideal', linewidth=2)
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Strong Scaling Factor', fontweight='bold')
    ax.set_title('Strong Scaling - Training Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    # Strong Scaling - Prediction
    ax = axes[0, 1]
    for trees in tree_counts:
        subset = df[df['NumTrees'] == trees].groupby('NumThreads')['PredictionTime_ms'].mean()
        baseline = subset.iloc[0]
        strong_scaling = baseline / subset
        ax.plot(subset.index, strong_scaling, marker='s', label=f'{trees} trees', linewidth=2)
    
    ax.plot([1, max_threads], [1, max_threads], 'k--', label='Ideal', linewidth=2)
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Strong Scaling Factor', fontweight='bold')
    ax.set_title('Strong Scaling - Prediction Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    # Throughput comparison
    ax = axes[1, 0]
    threads = sorted(df['NumThreads'].unique())
    train_throughput = df.groupby('NumThreads')['TrainingThroughput_samples_per_sec'].mean()
    pred_throughput = df.groupby('NumThreads')['PredictionThroughput_samples_per_sec'].mean()
    
    x = np.arange(len(threads))
    width = 0.35
    ax.bar(x - width/2, train_throughput.values, width, label='Training', alpha=0.8)
    ax.bar(x + width/2, pred_throughput.values, width, label='Prediction', alpha=0.8)
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Throughput (samples/sec)', fontweight='bold')
    ax.set_title('Throughput Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(threads)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Time breakdown
    ax = axes[1, 1]
    time_data = df.groupby('NumThreads')[['TrainingTime_ms', 'PredictionTime_ms']].mean()
    time_data.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Time Breakdown by Phase')
    ax.legend(['Training', 'Prediction'])
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'scalability_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_analysis(df, output_dir):
    """Plot accuracy metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Accuracy Analysis', fontsize=16, fontweight='bold')
    
    # Accuracy vs Threads
    ax = axes[0, 0]
    tree_counts = sorted(df['NumTrees'].unique())
    for trees in tree_counts:
        subset = df[df['NumTrees'] == trees].groupby('NumThreads')['Accuracy'].mean()
        ax.plot(subset.index, subset.values * 100, marker='o', label=f'{trees} trees', linewidth=2)
    
    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy vs Thread Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # F1 Score comparison
    ax = axes[0, 1]
    metrics = df.groupby('NumTrees')[['Accuracy', 'F1Score', 'Precision', 'Recall']].mean()
    metrics.plot(kind='bar', ax=ax, alpha=0.8)
    ax.set_xlabel('Number of Trees', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Classification Metrics by Tree Count')
    ax.legend(['Accuracy', 'F1 Score', 'Precision', 'Recall'])
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    # Accuracy vs Time tradeoff
    ax = axes[1, 0]
    ax.scatter(df['TotalTime_ms'], df['Accuracy'] * 100, 
              c=df['NumThreads'], cmap='viridis', s=100, alpha=0.6)
    ax.set_xlabel('Total Time (ms)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy vs Time Tradeoff')
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='Num Threads')
    ax.grid(True, alpha=0.3)
    
    # Memory usage
    ax = axes[1, 1]
    memory = df.groupby('NumTrees')['MemoryUsage_MB'].mean()
    ax.bar(range(len(memory)), memory.values, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(memory)))
    ax.set_xticklabels(memory.index)
    ax.set_xlabel('Number of Trees', fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontweight='bold')
    ax.set_title('Average Memory Usage by Tree Count')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'accuracy_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_report(df, output_dir):
    """Generate a text summary report"""
    output_path = Path(output_dir) / 'performance_summary.txt'
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE PERFORMANCE EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. CONFIGURATION SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of configurations tested: {len(df)}\n")
        f.write(f"Thread counts: {sorted(df['NumThreads'].unique())}\n")
        f.write(f"Tree counts: {sorted(df['NumTrees'].unique())}\n")
        f.write(f"Sample sizes: {sorted(df['SamplesPerTree'].unique())}\n\n")
        
        f.write("2. BEST PERFORMANCE CONFIGURATIONS\n")
        f.write("-" * 80 + "\n")
        
        # Best training performance
        best_train = df.loc[df['TrainingTime_ms'].idxmin()]
        f.write(f"Best Training Time: {best_train['TrainingTime_ms']:.2f} ms\n")
        f.write(f"  - Threads: {best_train['NumThreads']}, Trees: {best_train['NumTrees']}\n")
        f.write(f"  - Speedup: {best_train['TrainingSpeedup']:.2f}x, Efficiency: {best_train['TrainingEfficiency']*100:.1f}%\n\n")
        
        # Best prediction performance
        best_pred = df.loc[df['PredictionTime_ms'].idxmin()]
        f.write(f"Best Prediction Time: {best_pred['PredictionTime_ms']:.2f} ms\n")
        f.write(f"  - Threads: {best_pred['NumThreads']}, Trees: {best_pred['NumTrees']}\n")
        f.write(f"  - Speedup: {best_pred['PredictionSpeedup']:.2f}x, Efficiency: {best_pred['PredictionEfficiency']*100:.1f}%\n\n")
        
        # Best accuracy
        best_acc = df.loc[df['Accuracy'].idxmax()]
        f.write(f"Best Accuracy: {best_acc['Accuracy']*100:.2f}%\n")
        f.write(f"  - Threads: {best_acc['NumThreads']}, Trees: {best_acc['NumTrees']}\n")
        f.write(f"  - F1 Score: {best_acc['F1Score']:.4f}, Precision: {best_acc['Precision']:.4f}, Recall: {best_acc['Recall']:.4f}\n\n")
        
        f.write("3. SCALABILITY SUMMARY\n")
        f.write("-" * 80 + "\n")
        max_threads = df['NumThreads'].max()
        max_thread_data = df[df['NumThreads'] == max_threads]
        f.write(f"Average Training Speedup at {max_threads} threads: {max_thread_data['TrainingSpeedup'].mean():.2f}x\n")
        f.write(f"Average Training Efficiency at {max_threads} threads: {max_thread_data['TrainingEfficiency'].mean()*100:.1f}%\n")
        f.write(f"Average Prediction Speedup at {max_threads} threads: {max_thread_data['PredictionSpeedup'].mean():.2f}x\n")
        f.write(f"Average Prediction Efficiency at {max_threads} threads: {max_thread_data['PredictionEfficiency'].mean()*100:.1f}%\n\n")
        
        f.write("4. RESOURCE UTILIZATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Memory Usage: {df['MemoryUsage_MB'].mean():.2f} MB\n")
        f.write(f"Peak Memory Usage: {df['MemoryUsage_MB'].max():.2f} MB\n")
        f.write(f"Average Training Throughput: {df['TrainingThroughput_samples_per_sec'].mean():.2f} samples/sec\n")
        f.write(f"Average Prediction Throughput: {df['PredictionThroughput_samples_per_sec'].mean():.2f} samples/sec\n\n")
        
        f.write("5. RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        
        # Find optimal thread count (best efficiency > 70%)
        good_efficiency = df[df['OverallEfficiency'] > 0.7]
        if not good_efficiency.empty:
            optimal_threads = good_efficiency.groupby('NumThreads')['OverallSpeedup'].mean().idxmax()
            f.write(f"Recommended thread count: {optimal_threads}\n")
            f.write(f"  - Maintains >70% efficiency with good speedup\n")
        
        # Optimal tree count
        optimal_trees = df.groupby('NumTrees').apply(
            lambda x: (x['Accuracy'].mean(), x['TotalTime_ms'].mean())
        )
        f.write(f"\nOptimal configurations for different priorities:\n")
        f.write(f"  - For best accuracy: Use maximum trees tested\n")
        f.write(f"  - For balanced performance: Use medium tree count with high thread count\n")
        f.write(f"  - For fastest execution: Use fewer trees with optimal thread count\n")
        
    print(f"Saved summary report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze comprehensive performance results')
    parser.add_argument('csv_file', help='Path to comprehensive_performance.csv')
    parser.add_argument('--output-dir', default=None, help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load data
    df = load_comprehensive_results(args.csv_file)
    if df is None:
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.csv_file).parent / 'analysis'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_speedup_curves(df, output_dir, phase='Training')
    plot_speedup_curves(df, output_dir, phase='Prediction')
    plot_efficiency_curves(df, output_dir, phase='Training')
    plot_efficiency_curves(df, output_dir, phase='Prediction')
    plot_scalability_analysis(df, output_dir)
    plot_accuracy_analysis(df, output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete! Check the output directory for results.")
    print("="*80)


if __name__ == '__main__':
    main()
