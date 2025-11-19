#!/usr/bin/env python3
"""
MPI Performance Analysis and Visualization
Generates plots for MPI scalability, process vs threads analysis, and strong/weak scaling
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_mpi_results(csv_path):
    """Load MPI comprehensive performance results"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} configurations")
    print(f"Columns: {df.columns.tolist()}")
    return df


def plot_strong_scaling(df, output_dir):
    """Plot strong scaling efficiency for different configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Strong Scaling Analysis - MPI Implementation', fontsize=16, fontweight='bold')
    
    tree_counts = sorted(df['NumTrees'].unique())
    
    # Training speedup
    ax = axes[0, 0]
    for tree_count in tree_counts:
        subset = df[df['NumTrees'] == tree_count].sort_values('TotalWorkers')
        ax.plot(subset['TotalWorkers'], subset['TrainingSpeedup'], 
               marker='o', label=f'{tree_count} trees', linewidth=2, markersize=8)
    
    max_workers = df['TotalWorkers'].max()
    ax.plot([1, max_workers], [1, max_workers], 'k--', alpha=0.5, label='Ideal', linewidth=2)
    ax.set_xlabel('Total Workers (Processes × Threads)', fontweight='bold')
    ax.set_ylabel('Training Speedup', fontweight='bold')
    ax.set_title('Training Speedup vs Workers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    # Training efficiency
    ax = axes[0, 1]
    for tree_count in tree_counts:
        subset = df[df['NumTrees'] == tree_count].sort_values('TotalWorkers')
        ax.plot(subset['TotalWorkers'], subset['StrongScalingEfficiency'] * 100, 
               marker='s', label=f'{tree_count} trees', linewidth=2, markersize=8)
    
    ax.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Ideal (100%)', linewidth=2)
    ax.axhline(y=80, color='orange', linestyle=':', alpha=0.5, label='Good (80%)', linewidth=1.5)
    ax.set_xlabel('Total Workers', fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontweight='bold')
    ax.set_title('Strong Scaling Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # Prediction speedup
    ax = axes[1, 0]
    for tree_count in tree_counts:
        subset = df[df['NumTrees'] == tree_count].sort_values('TotalWorkers')
        ax.plot(subset['TotalWorkers'], subset['PredictionSpeedup'], 
               marker='^', label=f'{tree_count} trees', linewidth=2, markersize=8)
    
    ax.plot([1, max_workers], [1, max_workers], 'k--', alpha=0.5, label='Ideal', linewidth=2)
    ax.set_xlabel('Total Workers', fontweight='bold')
    ax.set_ylabel('Prediction Speedup', fontweight='bold')
    ax.set_title('Prediction Speedup vs Workers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    
    # Communication overhead
    ax = axes[1, 1]
    for proc_count in sorted(df['MPIProcesses'].unique()):
        subset = df[df['MPIProcesses'] == proc_count].sort_values('ThreadsPerProcess')
        avg_overhead = subset.groupby('ThreadsPerProcess')['CommunicationOverhead_ms'].mean()
        ax.plot(avg_overhead.index, avg_overhead.values, 
               marker='d', label=f'{proc_count} processes', linewidth=2, markersize=8)
    
    ax.set_xlabel('Threads per Process', fontweight='bold')
    ax.set_ylabel('Communication Overhead (ms)', fontweight='bold')
    ax.set_title('Communication Overhead vs Thread Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'mpi_strong_scaling.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_process_vs_threads(df, output_dir):
    """Analyze trade-off between MPI processes and threads per process"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MPI Processes vs Threads Per Process Analysis', fontsize=16, fontweight='bold')
    
    # Training time heatmap
    ax = axes[0, 0]
    pivot_train = df.pivot_table(values='TrainingTime_ms', 
                                 index='ThreadsPerProcess', 
                                 columns='MPIProcesses', 
                                 aggfunc='mean')
    im = sns.heatmap(pivot_train, annot=True, fmt='.0f', cmap='YlOrRd_r', ax=ax, cbar_kws={'label': 'Time (ms)'})
    ax.set_title('Training Time (ms)', fontweight='bold')
    ax.set_ylabel('Threads per Process', fontweight='bold')
    ax.set_xlabel('MPI Processes', fontweight='bold')
    
    # Prediction time heatmap
    ax = axes[0, 1]
    pivot_pred = df.pivot_table(values='PredictionTime_ms', 
                                index='ThreadsPerProcess', 
                                columns='MPIProcesses', 
                                aggfunc='mean')
    sns.heatmap(pivot_pred, annot=True, fmt='.0f', cmap='YlGnBu_r', ax=ax, cbar_kws={'label': 'Time (ms)'})
    ax.set_title('Prediction Time (ms)', fontweight='bold')
    ax.set_ylabel('Threads per Process', fontweight='bold')
    ax.set_xlabel('MPI Processes', fontweight='bold')
    
    # Efficiency heatmap
    ax = axes[1, 0]
    pivot_eff = df.pivot_table(values='StrongScalingEfficiency', 
                               index='ThreadsPerProcess', 
                               columns='MPIProcesses', 
                               aggfunc='mean')
    sns.heatmap(pivot_eff * 100, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax, 
                vmin=0, vmax=100, cbar_kws={'label': 'Efficiency (%)'})
    ax.set_title('Strong Scaling Efficiency (%)', fontweight='bold')
    ax.set_ylabel('Threads per Process', fontweight='bold')
    ax.set_xlabel('MPI Processes', fontweight='bold')
    
    # Total time for different configurations
    ax = axes[1, 1]
    tree_counts = sorted(df['NumTrees'].unique())
    for num_trees in tree_counts[:3]:  # Show first 3 for clarity
        subset = df[df['NumTrees'] == num_trees].sort_values('TotalWorkers')
        ax.plot(subset['TotalWorkers'], subset['TotalTime_ms'], 
               marker='o', label=f'{num_trees} trees', linewidth=2, markersize=8)
    
    ax.set_xlabel('Total Workers (Processes × Threads)', fontweight='bold')
    ax.set_ylabel('Total Time (ms)', fontweight='bold')
    ax.set_title('Total Execution Time vs Configuration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'mpi_process_thread_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_scalability_curves(df, output_dir):
    """Plot detailed scalability curves"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MPI Scalability Analysis', fontsize=16, fontweight='bold')
    
    # Strong scaling (fixed problem size)
    ax = axes[0, 0]
    sample_sizes = sorted(df['TrainSamples'].unique())
    for sample_size in sample_sizes[:3]:  # Show first 3 for clarity
        subset = df[df['TrainSamples'] == sample_size].sort_values('TotalWorkers')
        ax.plot(subset['TotalWorkers'], subset['TrainingTime_ms'], 
               marker='o', label=f'{sample_size} samples', linewidth=2, markersize=8)
    
    ax.set_xlabel('Total Workers', fontweight='bold')
    ax.set_ylabel('Training Time (ms)', fontweight='bold')
    ax.set_title('Strong Scaling: Training Time')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    # Weak scaling efficiency
    ax = axes[0, 1]
    weak_eff = df.groupby('TotalWorkers')['WeakScalingEfficiency'].mean() * 100
    ax.plot(weak_eff.index, weak_eff.values, marker='s', linewidth=2, markersize=10, color='steelblue')
    ax.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Ideal (100%)', linewidth=2)
    ax.axhline(y=80, color='orange', linestyle=':', alpha=0.5, label='Good (80%)', linewidth=1.5)
    ax.set_xlabel('Total Workers', fontweight='bold')
    ax.set_ylabel('Weak Scaling Efficiency (%)', fontweight='bold')
    ax.set_title('Weak Scaling Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # Throughput analysis
    ax = axes[1, 0]
    df['TrainingThroughput'] = (df['TrainSamples'] * df['NumTrees']) / (df['TrainingTime_ms'] / 1000)
    tree_counts = sorted(df['NumTrees'].unique())
    for num_trees in tree_counts:
        subset = df[df['NumTrees'] == num_trees].sort_values('TotalWorkers')
        throughput = subset.groupby('TotalWorkers')['TrainingThroughput'].mean()
        ax.plot(throughput.index, throughput.values, marker='^', 
               label=f'{num_trees} trees', linewidth=2, markersize=8)
    
    ax.set_xlabel('Total Workers', fontweight='bold')
    ax.set_ylabel('Throughput (samples/sec)', fontweight='bold')
    ax.set_title('Training Throughput')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # Memory usage per node
    ax = axes[1, 1]
    for proc_count in sorted(df['MPIProcesses'].unique()):
        subset = df[df['MPIProcesses'] == proc_count]
        mem_usage = subset.groupby('NumTrees')['MemoryPerNode_MB'].mean()
        ax.plot(mem_usage.index, mem_usage.values, marker='d', 
               label=f'{proc_count} processes', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Trees', fontweight='bold')
    ax.set_ylabel('Memory per Node (MB)', fontweight='bold')
    ax.set_title('Memory Usage per Node')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'mpi_scalability_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_metrics(df, output_dir):
    """Plot accuracy and quality metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MPI Accuracy and Quality Metrics', fontsize=16, fontweight='bold')
    
    # Accuracy vs configuration
    ax = axes[0, 0]
    for proc_count in sorted(df['MPIProcesses'].unique()):
        subset = df[df['MPIProcesses'] == proc_count].sort_values('TotalWorkers')
        acc = subset.groupby('TotalWorkers')['Accuracy'].mean() * 100
        ax.plot(acc.index, acc.values, marker='o', label=f'{proc_count} processes', linewidth=2)
    
    ax.set_xlabel('Total Workers', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy vs Worker Configuration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # F1 Score
    ax = axes[0, 1]
    metrics = df.groupby('NumTrees')[['Accuracy', 'F1Score', 'Precision', 'Recall']].mean()
    metrics.plot(kind='bar', ax=ax, alpha=0.8, width=0.7)
    ax.set_xlabel('Number of Trees', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Classification Metrics by Tree Count')
    ax.legend(['Accuracy', 'F1 Score', 'Precision', 'Recall'])
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    # Accuracy consistency
    ax = axes[1, 0]
    acc_std = df.groupby('TotalWorkers')['Accuracy'].std() * 100
    acc_mean = df.groupby('TotalWorkers')['Accuracy'].mean() * 100
    ax.errorbar(acc_mean.index, acc_mean.values, yerr=acc_std.values, 
                marker='s', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel('Total Workers', fontweight='bold')
    ax.set_ylabel('Accuracy (%) ± Std Dev', fontweight='bold')
    ax.set_title('Accuracy Consistency Across Configurations')
    ax.grid(True, alpha=0.3)
    
    # Time vs accuracy tradeoff
    ax = axes[1, 1]
    scatter = ax.scatter(df['TotalTime_ms'], df['Accuracy'] * 100, 
                        c=df['TotalWorkers'], cmap='viridis', s=100, alpha=0.6)
    ax.set_xlabel('Total Time (ms)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy vs Time Tradeoff')
    cbar = plt.colorbar(scatter, ax=ax, label='Total Workers')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'mpi_accuracy_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_report(df, output_path):
    """Generate a text summary report"""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MPI COMPREHENSIVE PERFORMANCE EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. CONFIGURATION SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total configurations tested: {len(df)}\n")
        f.write(f"MPI Processes tested: {sorted(df['MPIProcesses'].unique())}\n")
        f.write(f"Threads per process tested: {sorted(df['ThreadsPerProcess'].unique())}\n")
        f.write(f"Tree counts tested: {sorted(df['NumTrees'].unique())}\n")
        f.write(f"Sample sizes tested: {sorted(df['TrainSamples'].unique())}\n\n")
        
        f.write("2. BEST CONFIGURATIONS\n")
        f.write("-" * 80 + "\n")
        
        # Best training
        best_training = df.loc[df['TrainingTime_ms'].idxmin()]
        f.write(f"Best Training Performance:\n")
        f.write(f"  MPI Processes: {best_training['MPIProcesses']}\n")
        f.write(f"  Threads per Process: {best_training['ThreadsPerProcess']}\n")
        f.write(f"  Total Workers: {best_training['TotalWorkers']}\n")
        f.write(f"  Trees: {best_training['NumTrees']}\n")
        f.write(f"  Training Time: {best_training['TrainingTime_ms']:.2f} ms\n")
        f.write(f"  Training Speedup: {best_training['TrainingSpeedup']:.2f}x\n\n")
        
        # Best efficiency
        best_efficiency = df.loc[df['StrongScalingEfficiency'].idxmax()]
        f.write(f"Best Efficiency Configuration:\n")
        f.write(f"  MPI Processes: {best_efficiency['MPIProcesses']}\n")
        f.write(f"  Threads per Process: {best_efficiency['ThreadsPerProcess']}\n")
        f.write(f"  Total Workers: {best_efficiency['TotalWorkers']}\n")
        f.write(f"  Efficiency: {best_efficiency['StrongScalingEfficiency']*100:.2f}%\n")
        f.write(f"  Training Speedup: {best_efficiency['TrainingSpeedup']:.2f}x\n\n")
        
        # Best accuracy
        best_accuracy = df.loc[df['Accuracy'].idxmax()]
        f.write(f"Best Accuracy Configuration:\n")
        f.write(f"  Accuracy: {best_accuracy['Accuracy']*100:.2f}%\n")
        f.write(f"  F1 Score: {best_accuracy['F1Score']:.4f}\n")
        f.write(f"  Trees: {best_accuracy['NumTrees']}\n\n")
        
        f.write("3. SCALABILITY METRICS\n")
        f.write("-" * 80 + "\n")
        max_workers = df['TotalWorkers'].max()
        min_workers = df['TotalWorkers'].min()
        
        max_config = df[df['TotalWorkers'] == max_workers]
        min_config = df[df['TotalWorkers'] == min_workers]
        
        avg_train_speedup = max_config['TrainingSpeedup'].mean()
        avg_pred_speedup = max_config['PredictionSpeedup'].mean()
        avg_efficiency = max_config['StrongScalingEfficiency'].mean() * 100
        
        f.write(f"Workers Range: {min_workers} to {max_workers}\n")
        f.write(f"Average Training Speedup at {max_workers} workers: {avg_train_speedup:.2f}x\n")
        f.write(f"Average Prediction Speedup at {max_workers} workers: {avg_pred_speedup:.2f}x\n")
        f.write(f"Average Strong Scaling Efficiency at {max_workers} workers: {avg_efficiency:.2f}%\n\n")
        
        f.write("4. COMMUNICATION OVERHEAD\n")
        f.write("-" * 80 + "\n")
        avg_overhead = df.groupby('MPIProcesses')['CommunicationOverhead_ms'].mean()
        for proc_count, overhead in avg_overhead.items():
            pct = (overhead / df[df['MPIProcesses']==proc_count]['TotalTime_ms'].mean()) * 100
            f.write(f"  {proc_count} processes: {overhead:.2f} ms ({pct:.2f}% of total time)\n")
        f.write("\n")
        
        f.write("5. ACCURACY SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Accuracy: {df['Accuracy'].mean()*100:.2f}%\n")
        f.write(f"Average F1-Score: {df['F1Score'].mean():.4f}\n")
        f.write(f"Average Precision: {df['Precision'].mean():.4f}\n")
        f.write(f"Average Recall: {df['Recall'].mean():.4f}\n")
        f.write(f"Accuracy Std Dev: {df['Accuracy'].std()*100:.2f}%\n\n")
        
        f.write("6. RESOURCE UTILIZATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Memory per Node: {df['MemoryPerNode_MB'].mean():.2f} MB\n")
        f.write(f"Peak Memory per Node: {df['MemoryPerNode_MB'].max():.2f} MB\n")
        f.write(f"Average Training Throughput: {df['TrainingThroughput'].mean():.2f} samples/sec\n\n")
        
        f.write("7. RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        
        # Efficiency analysis
        if avg_efficiency < 70:
            f.write("⚠ Low parallel efficiency detected (<70%)\n")
            f.write("  Recommendations:\n")
            f.write("  - Increase problem size per worker\n")
            f.write("  - Reduce MPI processes, increase threads per process\n")
            f.write("  - Optimize communication patterns\n")
            f.write("  - Consider larger tree counts\n\n")
        elif avg_efficiency > 85:
            f.write("✓ Excellent parallel efficiency (>85%)\n")
            f.write("  The implementation scales well. Consider:\n")
            f.write("  - Increasing worker count further if available\n")
            f.write("  - Testing with larger datasets\n\n")
        
        # Communication overhead analysis
        avg_comm_pct = (df['CommunicationOverhead_ms'] / df['TotalTime_ms']).mean() * 100
        if avg_comm_pct > 20:
            f.write("⚠ High communication overhead (>20% of total time)\n")
            f.write("  Recommendations:\n")
            f.write("  - Batch MPI communications\n")
            f.write("  - Use non-blocking sends/receives\n")
            f.write("  - Increase work per communication\n")
            f.write("  - Consider fewer MPI processes with more threads\n\n")
        
        # Optimal configuration
        f.write("OPTIMAL CONFIGURATION SUGGESTIONS:\n")
        f.write("-" * 80 + "\n")
        
        # Find sweet spot (good efficiency, good speedup)
        good_configs = df[df['StrongScalingEfficiency'] > 0.75].sort_values('TrainingSpeedup', ascending=False)
        if not good_configs.empty:
            optimal = good_configs.iloc[0]
            f.write(f"Balanced Configuration (>75% efficiency, best speedup):\n")
            f.write(f"  MPI Processes: {optimal['MPIProcesses']}\n")
            f.write(f"  Threads per Process: {optimal['ThreadsPerProcess']}\n")
            f.write(f"  Training Speedup: {optimal['TrainingSpeedup']:.2f}x\n")
            f.write(f"  Efficiency: {optimal['StrongScalingEfficiency']*100:.2f}%\n")
    
    print(f"Saved summary report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze MPI comprehensive performance results')
    parser.add_argument('csv_file', nargs='?', 
                       default='../results/mpi/mpi_comprehensive_performance.csv',
                       help='Path to mpi_comprehensive_performance.csv')
    parser.add_argument('--output-dir', default=None, help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load data
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        print(f"Please run MPI evaluation first or specify correct path.")
        return
    
    df = load_mpi_results(csv_path)
    if df is None or len(df) == 0:
        print("Error: No data loaded")
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = csv_path.parent / 'analysis'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Generate plots
    print("\nGenerating MPI analysis plots...")
    plot_strong_scaling(df, output_dir)
    plot_process_vs_threads(df, output_dir)
    plot_scalability_curves(df, output_dir)
    plot_accuracy_metrics(df, output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, output_dir / 'mpi_performance_summary.txt')
    
    print("\n" + "="*80)
    print("MPI Analysis complete! Check the output directory for results.")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - mpi_strong_scaling.png")
    print(f"  - mpi_process_thread_analysis.png")
    print(f"  - mpi_scalability_curves.png")
    print(f"  - mpi_accuracy_analysis.png")
    print(f"  - mpi_performance_summary.txt")


if __name__ == "__main__":
    main()
