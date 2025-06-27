import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil

from pathlib import Path


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
    df_merged['Speedup'] = np.maximum(df_merged['TrainingTime_ms_first'],  df_merged['TrainingTime_ms_second']) / np.minimum(df_merged['TrainingTime_ms_first'],  df_merged['TrainingTime_ms_second'])
    df_merged['Accuracy_Diff'] = df_merged['Accuracy_first'] - df_merged['Accuracy_second']
    df_merged['F1_Score_Diff'] = df_merged['F1Score_first'] - df_merged['F1Score_second']
    df_merged['Memory_Usage_Diff_MB'] = df_merged['MemoryUsage_MB_first'] - df_merged['MemoryUsage_MB_second']

    print("--- Performance Comparison ---")
    print(df_merged[[
        'SamplesPerTree', 'NumTrees', 'TrainingTime_ms_first', 'TrainingTime_ms_second', 'Speedup',
        'Accuracy_Diff', 'F1_Score_Diff', 'Memory_Usage_Diff_MB'
    ]].round(2))
    print("-" * 30)

    samples_per_tree = sorted(df_merged['SamplesPerTree'].unique())

    for samples in samples_per_tree:
        subset = df_merged[df_merged['SamplesPerTree'] == samples].copy()
        subset['Config'] = subset['NumTrees'].astype(str)

        plt.figure(figsize=(10, 6))
        plt.plot(subset['NumTrees'], subset['TrainingTime_ms_first'], label=f'{first_name}', marker='o', linestyle='--')
        plt.plot(subset['NumTrees'], subset['TrainingTime_ms_second'], label=f'{second_name}', marker='x', linestyle='-')
        plt.xlabel('Number of Trees')
        plt.ylabel('Training Time (ms)')
        plt.title(f'Training Time Comparison for SamplesPerTree = {samples}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path_to_save / f'training_time_comparison_samples_{samples}.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(subset['NumTrees'], subset['Speedup'], marker='^', linestyle='-', color='purple')
        plt.xlabel('Number of Trees')
        plt.ylabel('Speedup Factor')
        plt.title('Speedup Factor Comparison')
        plt.axhline(y=num_cores, color='r', linestyle=':', label=f'Ideal Speedup ({num_cores} cores)')

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_to_save / 'speedup_factor_comparison.png')
        plt.close()

        plt.figure(figsize=(12,7))
        colors = ['green' if x >= 0 else 'red' for x in subset['F1_Score_Diff']]
        plt.bar(subset['Config'], subset['F1_Score_Diff'], color=colors)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xlabel('Number of Trees')
        plt.ylabel(f'F1 Score Difference ({first_name} - {second_name})')
        plt.title(f'F1 Score Difference for SamplesPerTree = {samples} (Red: {first_name} < {second_name}, Green: {first_name} > {second_name})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(path_to_save / f'f1_score_difference_samples_{samples}.png')
        plt.close()

        plt.figure(figsize=(12,7))
        plt.figure(figsize=(12, 7))
        plt.plot(subset['NumTrees'], subset['MemoryUsage_MB_first'], marker='o', linestyle='--', label=f'{first_name} Memory (MB)')
        plt.plot(subset['NumTrees'], subset['MemoryUsage_MB_second'], marker='s', linestyle='-', label=f'{second_name} Memory (MB)')
        plt.title(f'Memory Usage Comparison for {samples} Samples/Tree ({first_name} vs {second_name})')
        plt.xlabel('Number of Trees')
        plt.ylabel(f'Memory Usage (MB) - {first_name} vs {second_name}')
        plt.grid(True)
        plt.legend()
        plt.savefig(path_to_save / f'memory_usage_comparison_samples_{samples}.png')
        plt.close()
        

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_performance.py <path1> <path2> ...")
    elif len(sys.argv) == 3:
        analyze_performance(sys.argv[1], sys.argv[2])
    else:
        print("\n\nPlease provide exactly two CSV file paths for comparison.\n\n")
        sys.exit(1)
        
        