import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil

from pathlib import Path
from matplotlib.ticker import FuncFormatter


def adaptive_time(x):
    """Format a time in seconds into ms/s/min/h as appropriate."""
    if x >= 3600:
        return f"{x/3600:.1f}h"
    elif x >= 60:
        return f"{x/60:.1f}m"
    elif x >= 1:
        return f"{x:.2f}s"
    else:
        return f"{x*1000:.0f}ms"

def analyze_performance(firs_csv: str, scnd_csv: str) -> None:
    """
    Analyze performance by comparing two CSV files containing performance data.
    
    Args:
        firs_csv (str): Path to the first CSV file.
        scnd_csv (str): Path to the second CSV file.
    """
    if not os.path.exists(firs_csv):
        raise FileNotFoundError(f"File {firs_csv} does not exist.")
    if not os.path.exists(scnd_csv):
        raise FileNotFoundError(f"File {scnd_csv} does not exist.")
    an_path = Path(firs_csv).parent.parent / 'analysis'
    if not os.path.exists(an_path ):
        os.makedirs(an_path, exist_ok=True)
    num_cores = os.cpu_count()
    if num_cores is None:
        num_cores = 1

    print(f"Plots will be saved in {an_path}")
    
    # Load the data from the CSV files
    first_df = pd.read_csv(firs_csv)
    second_df = pd.read_csv(scnd_csv)

    
    # Ensure both DataFrames have the same columns
    if not all(first_df.columns == second_df.columns):
        raise ValueError("CSV files must have the same columns.")

    first_name = Path(firs_csv).parent.stem
    second_name = Path(scnd_csv).parent.stem
    print(f"\nAnalyzing performance between {first_name} and {second_name}")
    path_to_save = an_path / f"{first_name}_vs_{second_name}"
    if path_to_save.exists():
        shutil.rmtree(path_to_save)
    os.makedirs(path_to_save)
    os.makedirs(path_to_save, exist_ok=True)
    
    df_merged = pd.merge(first_df, second_df, on=['NumTrees', 'SamplesPerTree'], suffixes=('_first', '_second'))
    df_merged['Train_Speedup']      = np.maximum(df_merged['TrainingTime_ms_first'], df_merged['TrainingTime_ms_second'])  / np.minimum(df_merged['TrainingTime_ms_first'], df_merged['TrainingTime_ms_second'])
    df_merged['Pred_Speedup']       = np.maximum(df_merged['PredictionTime_ms_first'], df_merged['PredictionTime_ms_second']) / np.minimum(df_merged['PredictionTime_ms_first'], df_merged['PredictionTime_ms_second'])
    df_merged['Parallel_Efficiency'] = df_merged['Train_Speedup'] / num_cores


    df_merged['TrainingTime_ms_first']   /= 1000
    df_merged['TrainingTime_ms_second']  /= 1000

    df_merged['PredictionTime_ms_first']  /= 1000
    df_merged['PredictionTime_ms_second'] /= 1000

    # rename all four
    df_merged.rename(columns={
        'TrainingTime_ms_first':    'TrainingTime_s_first',
        'TrainingTime_ms_second':   'TrainingTime_s_second',
        'PredictionTime_ms_first':  'PredictionTime_s_first',
        'PredictionTime_ms_second': 'PredictionTime_s_second'
    }, inplace=True)


    df_merged['Accuracy_Diff'] = df_merged['Accuracy_first'] - df_merged['Accuracy_second']
    df_merged['F1_Score_Diff'] = df_merged['F1Score_first'] - df_merged['F1Score_second']
    df_merged['Memory_Usage_Diff_MB'] = df_merged['MemoryUsage_MB_first'] - df_merged['MemoryUsage_MB_second']
    df_merged['MemoryPerTree_diff_MB'] = df_merged['Memory_Usage_Diff_MB'] / df_merged['NumTrees']
    
    df_merged['Throughput_Speedup'] = (
        df_merged['TrainingThroughput_samples_per_sec_second'] /
        df_merged['TrainingThroughput_samples_per_sec_first']
    )

    print("--- Performance Comparison ---")
    print(df_merged[[
        'SamplesPerTree', 'NumTrees', 'TrainingTime_s_first', 'TrainingTime_s_second', 'Train_Speedup', 'Pred_Speedup', 'Parallel_Efficiency',
        'Accuracy_Diff', 'F1_Score_Diff', 'Memory_Usage_Diff_MB'
    ]].round(2))
    print("-" * 30)
    
    for samples in sorted(df_merged['SamplesPerTree'].unique()):
        subset = df_merged[df_merged['SamplesPerTree']==samples]

        # Plot 1: Time, Speedup, and Memory
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
        fig.suptitle(f'Performance Analysis (Samples per Tree = {samples})', fontsize=16)
        ax1, ax2, ax3, ax4 = axes.flatten()

        # 1) Training time
        ax1.plot(subset['NumTrees'], subset['TrainingTime_s_first'], marker='o', linestyle='--', label=first_name)
        ax1.plot(subset['NumTrees'], subset['TrainingTime_s_second'], marker='x', linestyle='-',  label=second_name)
        ax1.set_title('Training Time')
        ax1.set_ylabel('Time')
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda v,p: adaptive_time(v)))
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 2) Prediction time
        ax2.plot(subset['NumTrees'], subset['PredictionTime_s_first'], marker='o', linestyle='--', label=first_name)
        ax2.plot(subset['NumTrees'], subset['PredictionTime_s_second'], marker='x', linestyle='-',  label=second_name)
        ax2.set_title('Prediction Time')
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v,p: adaptive_time(v)))
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        # 3) Speedup
        ax3.plot(subset['NumTrees'], subset['Train_Speedup'], marker='^', color='purple', label='Train Speedup')
        ax3.plot(subset['NumTrees'], subset['Pred_Speedup'], marker='s', color='teal', label='Pred Speedup')
        ax3.axhline(num_cores, color='r', linestyle=':', label=f'Ideal Speedup ({num_cores} cores)')
        ax3.set_title('Speedup')
        ax3.set_ylabel('Speedup Factor')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.6)

        # 4) Memory Usage
        ax4.plot(subset['NumTrees'], subset['MemoryUsage_MB_first'], marker='o', linestyle='--', label=first_name)
        ax4.plot(subset['NumTrees'], subset['MemoryUsage_MB_second'], marker='x', linestyle='-', label=second_name)
        ax4.set_title('Memory Usage')
        ax4.set_ylabel('Memory (MB)')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.6)

        for ax in axes[-1]:
            ax.set_xlabel('Number of Trees')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path_to_save / f'performance_overview_{samples}.png')
        plt.close()
        print(f"Saved performance_overview_{samples}.png")

        # Plot 2: Throughput, Accuracy, and Efficiency
        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
        fig2.suptitle(f'Detailed Metrics (Samples per Tree = {samples})', fontsize=16)
        ax5, ax6, ax7, ax8 = axes2.flatten()

        # 5) Throughput
        ax5.plot(subset['NumTrees'], subset['TrainingThroughput_samples_per_sec_first'], marker='o', label=f'{first_name} Train')
        ax5.plot(subset['NumTrees'], subset['TrainingThroughput_samples_per_sec_second'], marker='x', label=f'{second_name} Train')
        ax5.set_title('Training Throughput')
        ax5.set_ylabel('Samples / sec')
        ax5.legend()
        ax5.grid(True, linestyle='--', alpha=0.6)

        # 6) Accuracy & F1 Score
        ax6.plot(subset['NumTrees'], subset['Accuracy_first'], marker='o', linestyle='-', label=f'{first_name} Accuracy')
        ax6.plot(subset['NumTrees'], subset['Accuracy_second'], marker='x', linestyle='--', label=f'{second_name} Accuracy')
        ax6.plot(subset['NumTrees'], subset['F1Score_first'], marker='s', linestyle='-', label=f'{first_name} F1')
        ax6.plot(subset['NumTrees'], subset['F1Score_second'], marker='d', linestyle='--', label=f'{second_name} F1')
        ax6.set_title('Model Quality')
        ax6.set_ylabel('Score')
        ax6.set_ylim(0, 1)
        ax6.legend()
        ax6.grid(True, linestyle='--', alpha=0.6)

        # 7) Parallel Efficiency
        ax7.plot(subset['NumTrees'], subset['Parallel_Efficiency'], marker='p', color='green', label='Parallel Efficiency')
        ax7.axhline(1.0, color='r', linestyle=':', label='Ideal Efficiency (1.0)')
        ax7.set_title('Parallel Efficiency (Training)')
        ax7.set_ylabel('Efficiency')
        ax7.set_ylim(bottom=0)
        ax7.legend()
        ax7.grid(True, linestyle='--', alpha=0.6)

        # 8) Memory-per-tree diff
        ax8.bar(subset['NumTrees'], subset['MemoryPerTree_diff_MB'], color='cornflowerblue')
        ax8.axhline(0, color='black', linestyle='--', lw=0.8)
        ax8.set_title('Memory Difference per Tree')
        ax8.set_ylabel(f'Δ Memory per Tree (MB)\n({first_name} - {second_name})')
        ax8.grid(True, linestyle='--', alpha=0.6)

        for ax in axes2[-1]:
            ax.set_xlabel('Number of Trees')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path_to_save / f'detailed_metrics_{samples}.png')
        plt.close()
        print(f"Saved detailed_metrics_{samples}.png")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_performance.py <path1> <path2> ...")
    elif len(sys.argv) == 3:
        analyze_performance(sys.argv[1], sys.argv[2])
    else:
        print("\n\nPlease provide exactly two CSV file paths for comparison.\n\n")
        sys.exit(1)
        
        