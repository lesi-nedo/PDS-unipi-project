import pandas as pd
import sys
import os

from pathlib import Path
from sklearn.model_selection import train_test_split

from utils import load_dataset

def preprocess_data(file_path="../../../data/iris/iris.data", test_size=0.2, random_state=42):
    """
    Preprocess the dataset by loading it, splitting into training and test sets,
    and encoding the labels.

    Parameters:
    file_path (str): Path to the CSV file.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: Training features (X_train), training labels (y_train),
           test features (X_test), test labels (y_test), and label encoder.
    """
    X, y, _ = load_dataset(file_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    test_data = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
    test_data['label'] = y_test

    train_data = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    train_data['label'] = y_train

    path_to_save = Path(file_path).parent.absolute()
    train_file = path_to_save / f"train_{path_to_save.name}.csv"
    test_file = path_to_save / f"test_{path_to_save.name}.csv"
    train_data.to_csv(train_file, index=False, header=False)
    test_data.to_csv(test_file, index=False, header=False)
    paths_file = Path(__file__).parent / "iris_paths.txt"
    with open(paths_file, 'w') as f:
        f.write(f"{train_file}\n{test_file}\n")

    return str(train_file),str(test_file)

if __name__ == "__main__":
    try:
        
        preprocess_data()
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}", file=sys.stderr)
        sys.exit(1)
    print("Data preprocessing complete. Train and test data saved to CSV files.")

