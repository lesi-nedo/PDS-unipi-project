import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


def compare_impl(train_data_path, test_data_path):
    # Load the training and testing data
    train_data = pd.read_csv(train_data_path, header=None)
    test_data = pd.read_csv(test_data_path, header=None)
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    print("\n" + "=" * 50)
    print("Running Sklearn Random Forest Classifier")
    print("=" * 50 + "\n")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    sklearn_accuracy = accuracy_score(y_test, y_pred)
    sklearn_f1 = f1_score(y_test, y_pred, average='weighted')
    

    np.savetxt("../../../results/validation/seq_sklearn/sklearn_predictions.csv",
               y_pred, delimiter=",", fmt="%d")
    np.savetxt("../../../results/validation/seq_sklearn/sklearn_probabilities.csv",
               y_pred_proba, delimiter=",", fmt="%.4f")
    
    try:
        cpp_predictions = np.loadtxt("../../../results/validation/seq_sklearn/cpp_predictions.txt", dtype=int)
        cpp_probabilities = np.loadtxt("../../../results/validation/seq_sklearn/cpp_probabilities.csv",
                                       delimiter=",")
        
        cpp_accuracy = accuracy_score(y_test, cpp_predictions)
        cpp_f1 = f1_score(y_test, cpp_predictions, average='weighted')
        
        print("\n" + "=" * 50)
        print("\nComparing with C++ implementation results:")
        print("=" * 50 + "\n")


        print("\nDetailed comparison (first 10 samples):")
        print("Sample | Actual | C++ Pred | Sklearn Pred | Agreement")
        print("-------|--------|----------|--------------|----------")
        
        for i in range(min(10, len(y_test))):
            agreement = "✓" if cpp_predictions[i] == y_pred[i] else "✗"
            print(f"{i:6d} | {y_test[i]:6d} | {cpp_predictions[i]:8d} | {y_pred[i]:12d} | {agreement:9s}")

        print("\n" + "=" * 50)
        print(f"Statistics Summary:")
        print(f"Total Samples: {len(y_test)}")
        prediction_aggreement = np.mean(y_pred == cpp_predictions)
        print("Prediction Agreement: {:.4f}".format(prediction_aggreement))

        prob_diff = np.mean(np.abs(y_pred_proba - cpp_probabilities))
        print("Mean Probability Difference: {:.4f}".format(prob_diff))


        print("Sklearn Random Forest Classifier Accuracy: {:.4f}".format(sklearn_accuracy))
        print("Sklearn Random Forest Classifier F1 Score: {:.4f}".format(sklearn_f1))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        print("C++ Predictions Accuracy: {:.4f}".format(
            cpp_accuracy))
        print("C++ Predictions F1 Score: {:.4f}".format(
            cpp_f1))
        print("\nClassification Report for C++ Predictions:\n",
              classification_report(y_test, cpp_predictions))
    except FileNotFoundError:
        print("\n" + "=" * 50)
        print("C++ implementation results not found. Skipping comparison.")
        print("=" * 50 + "\n")

    try:
        print("\n" + "=" * 50)
        print("Saving plots")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        output_dir = "../../../results/validation/seq_sklearn/plots"
        import os
        os.makedirs(output_dir, exist_ok=True)

         # Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        cm_sklearn = confusion_matrix(y_test, y_pred)
        cm_cpp = confusion_matrix(y_test, cpp_predictions)
        sns.heatmap(cm_sklearn, annot=True, fmt="d", ax=axes[0], cmap="Blues")
        axes[0].set_title("Sklearn Confusion Matrix")
        sns.heatmap(cm_cpp, annot=True, fmt="d", ax=axes[1], cmap="Greens")
        axes[1].set_title("C++ Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrices.png"))
        plt.close() 

        # Bar plot: Accuracy and F1 Score
        metrics = ['Accuracy', 'F1 Score']
        sklearn_scores = [sklearn_accuracy, sklearn_f1]
        cpp_scores = [
            cpp_accuracy,
            cpp_f1
        ]
        x = np.arange(len(metrics))
        width = 0.35

        plt.figure(figsize=(6, 4))
        plt.bar(x - width/2, sklearn_scores, width, label='Sklearn', color='royalblue')
        plt.bar(x + width/2, cpp_scores, width, label='C++', color='seagreen')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(x, metrics)
        plt.title('Validation Metrics Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
        plt.close()

        # Histogram: Probability Differences
        plt.figure(figsize=(6, 4))
        sns.histplot(np.abs(y_pred_proba - cpp_probabilities).flatten(), bins=30, kde=True, color="orange")
        plt.title("Distribution of Probability Differences")
        plt.xlabel("Absolute Difference")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "probability_difference_hist.png"))
        plt.close()

    except ImportError:
        print("\n" + "=" * 50)
        print("Matplotlib or Seaborn not installed. Skipping plot generation.")
        print("=" * 50 + "\n")
    print("\n" + "=" * 50)
    print("Sklearn Random Forest Classifier completed.")
    print("=" * 50 + "\n")
    return sklearn_accuracy, sklearn_f1
if __name__ == "__main__":
    import os
    import sys

    
    print("Current Working Directory:", os.getcwd())
    if len(sys.argv) != 3:
        print("Usage: python validation.py <train_data_path> <test_data_path>")
        sys.exit(1)
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    if not os.path.exists(train_data_path):
        print(f"Training data file '{train_data_path}' does not exist.")
        sys.exit(1)
    if not os.path.exists(test_data_path):
        print(f"Testing data file '{test_data_path}' does not exist.")
        sys.exit(1)
    print("Training data path:", train_data_path)
    print("Testing data path:", test_data_path)
    compare_impl(train_data_path, test_data_path)




