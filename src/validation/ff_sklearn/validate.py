import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
        
def compare_impl(train_data_path, test_data_path, ff_rf_pred_path):
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print("Error: Training or testing data file does not exist.")
        return
    if not os.path.exists(ff_rf_pred_path):
        print("Error: FastFlow Random Forest predictions folder does not exist.")
        return
    path_with_files = "";
    with open(ff_rf_pred_path + "/path.txt", "r") as f:
        path_with_files = f.read().strip()
    if not os.path.exists(path_with_files):
        print(f"Error: The path {path_with_files} does not exist. Please check the FastFlow predictions path.")
        return
    if not os.path.isdir(ff_rf_pred_path):
        print(f"Error: The path {ff_rf_pred_path} is not a directory. Please provide a valid directory for FastFlow predictions.")
        return
    ff_rf_pred_path = path_with_files
    samples_train_set = set();
    samples_test_set = set();

    tree_size_set = set();
    for file in os.listdir(ff_rf_pred_path):
        if file.endswith(".csv"):
            file_without_ext = os.path.splitext(file)[0]
            if file_without_ext.startswith("ff_predictions_"):
                file_parts = file_without_ext.split('_')
                
                tree_size_set.add(int(file_parts[2]))
                samples_train_set.add(int(file_parts[4]))
                samples_test_set.add(int(file_parts[5]))
                


    print("\n" + "=" * 50)
    print("Running Sklearn Random Forest Classifier")
    print("=" * 50 + "\n")
    
    print(f"Train Samples: {sorted(samples_train_set)}")
    print(f"Test Samples: {sorted(samples_test_set)}")
    print(f"Tree sizes: {sorted(tree_size_set)}")
    print("\n")

    for samples in zip(sorted(samples_train_set), sorted(samples_test_set)):
        for tree_size in sorted(tree_size_set):
            print(f"Processing samples: {samples}, tree size: {tree_size}")
            # Load the training and testing data
            train_data = pd.read_csv(train_data_path, header=None, nrows=samples[0])
            test_data = pd.read_csv(test_data_path, header=None, nrows=samples[1])
            X_train = train_data.iloc[:, :-1].values
            y_train = train_data.iloc[:, -1].values
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            rf = RandomForestClassifier(n_estimators=tree_size, random_state=42)
            start = time.perf_counter()
            rf.fit(X_train, y_train)
            end = time.perf_counter()
            print(f"Training time for {tree_size} trees with {samples[0]} training samples: {end - start:.4f} seconds")
            y_pred = rf.predict(X_test)
            y_pred_proba = rf.predict_proba(X_test)
            sklearn_accuracy = accuracy_score(y_test, y_pred)
            sklearn_f1 = f1_score(y_test, y_pred, average='weighted')

            np.savetxt(ff_rf_pred_path + f"/sklearn_predictions_{tree_size}_trees_{samples[0]}_{samples[1]}_samples.csv",
                       y_pred, delimiter=",", fmt="%d")
            np.savetxt(ff_rf_pred_path + f"/sklearn_probabilities_{tree_size}_trees_{samples[0]}_{samples[1]}_samples.csv",
                       y_pred_proba, delimiter=",", fmt="%.4f")
            
            print("\n" + "=" * 50)
            print("\nComparing with FastFlow implementation results:")
            print("=" * 50 + "\n")

            print("\nDetailed comparison (first 10 samples):")
            print("Sample | Actual | FastFlow Pred | Sklearn Pred | Agreement")
            print("-------|--------|----------------|--------------|----------")

            try:
                cpp_pred = np.loadtxt(
                    ff_rf_pred_path + f"/ff_predictions_{tree_size}_trees_{samples[0]}_{samples[1]}_samples.csv",
                    delimiter=",", dtype=int)
                cpp_prob = np.loadtxt(
                    ff_rf_pred_path + f"/ff_probabilities_{tree_size}_trees_{samples[0]}_{samples[1]}_samples.csv",
                    delimiter=",")
                
                print(f"Length: {len(y_test)}--- C++ Predictions: {len(cpp_pred)} -- C++ Probabilities: {cpp_prob.shape}")
                cpp_accuracy = accuracy_score(y_test, cpp_pred)
                cpp_f1 = f1_score(y_test, cpp_pred, average='weighted')

                for i in range(min(10, len(y_test))):
                    agreement = "✓" if cpp_pred[i] == y_pred[i] else "✗"
                    print(f"{i:6d} | {int(y_test[i]):6d} | {int(cpp_pred[i]):14d} | {int(y_pred[i]):12d} | {agreement:9s}")
                print("\n" + "=" * 50)
                print(f"Statistics Summary:")
                print(f"Total Samples: {len(y_test)}")
                prediction_agreement = np.mean(y_pred == cpp_pred)
                print("Prediction Agreement: {:.4f}".format(prediction_agreement))
                prob_diff = np.mean(np.abs(y_pred_proba - cpp_prob))
                print("Mean Probability Difference: {:.4f}".format(prob_diff))

                print("Sklearn Random Forest Classifier Accuracy: {:.4f}".format(sklearn_accuracy))
                print("Sklearn Random Forest Classifier F1 Score: {:.4f}".format(sklearn_f1))
                print("FastFlow Random Forest Classifier Accuracy: {:.4f}".format(cpp_accuracy))
                print("FastFlow Random Forest Classifier F1 Score: {:.4f}".format(cpp_f1))

                print("\nClassification Report for C++ implementation:\n", classification_report(y_test, cpp_pred))
                print("\nClassification Report for Sklearn implementation:\n", classification_report(y_test, y_pred))
            except FileNotFoundError as e:
                print(f"Error: {e}. Make sure the FastFlow predictions are generated correctly for samples: {samples}, tree size: {tree_size}.")
                raise
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise


            print("\n" + "=" * 50)
            try:
                output_dir = ff_rf_pred_path + f"/plots_{tree_size}_trees_{samples}_samples"
                print(f"Saving plots for samples: {samples}, tree size: {tree_size}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                print(f"Plots saved in {output_dir}")
            except Exception as e:
                print(f"Error saving plots: {e}")
                exit(1)

            # Plot confusion matrix
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            cm_sklearn = confusion_matrix(y_test, y_pred)
            cm_cpp = confusion_matrix(y_test, cpp_pred)
            sns.heatmap(cm_sklearn, annot=True, fmt='d', ax=axes[0], cmap='Blues')
            axes[0].set_title('Sklearn Random Forest Confusion Matrix')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
            sns.heatmap(cm_cpp, annot=True, fmt='d', ax=axes[1], cmap='Blues')
            axes[1].set_title('FastFlow Random Forest Confusion Matrix')
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Actual')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/confusion_matrices_{tree_size}_trees_{samples}_samples.png")
            plt.close(fig)

            # Bar plot: Accuracy and F1 Score
            metrics = ['Accuracy', 'F1 Score']
            sklearn_metrics = [sklearn_accuracy, sklearn_f1]
            cpp_metrics = [cpp_accuracy, cpp_f1]
            x = np.arange(len(metrics))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.bar(x - width/2, sklearn_metrics, width, label='Sklearn', color='blue')
            plt.bar(x + width/2, cpp_metrics, width, label='FastFlow', color='orange')
            ax.set_ylabel('Scores')
            ax.set_title(f'Accuracy and F1 Score Comparison (Samples: {samples}, Trees: {tree_size})')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/accuracy_f1_comparison_{tree_size}_trees_{samples}_samples.png")
            plt.close(fig)

            # Histogram of probabilities Difference
            fig, ax = plt.subplots(figsize=(8, 6))
            prob_diff = np.abs(y_pred_proba - cpp_prob)
            ax.hist(prob_diff.flatten(), bins=50, color='purple', alpha=0.7)
            ax.set_title(f'Probability Difference Histogram (Samples: {samples}, Trees: {tree_size})')
            ax.set_xlabel('Probability Difference')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/probability_difference_hist_{tree_size}_trees_{samples}_samples.png")
            plt.close(fig)

            print("\n" + "=" * 50)
            print("Plots saved successfully.")


            





if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python validate.py <train_data_path> <test_data_path> <ff_rf_pred_path>")
        sys.exit(1)

    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    ff_rf_pred_path = sys.argv[3]

    compare_impl(train_data_path, test_data_path, ff_rf_pred_path)
