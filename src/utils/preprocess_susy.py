import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_dataset(file_path):
    """
    Load dataset from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    tuple: Features (X), labels (y).
    """
    try:
        data = pd.read_csv(file_path, header=None)
        x = data.iloc[:, 1:-1].values
        y = data.iloc[:, 0].values

        return x, y
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {file_path}")
    except pd.errors.ParserError:
        raise ValueError(f"Error parsing file: {file_path}")
    

def preprocess_data(file_path="../../data/susy/SUSY.csv", test_size=0.2, random_state=None):
    """
    Preprocess the dataset by loading it, splitting into training and test sets.

    Parameters:
    file_path (str): Path to the CSV file.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: Training features (X_train), training labels (y_train),
           test features (X_test), test labels (y_test).
    """
    X, y = load_dataset(file_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    path_to_save = Path(file_path).parent.absolute()
    train_file = path_to_save / f"train_{path_to_save.name}.csv"
    test_file = path_to_save / f"test_{path_to_save.name}.csv"

    train_data = pd.DataFrame(X_train)
    train_data[len(train_data.columns)] = y_train
    train_data.to_csv(train_file, index=False, header=False)
    test_data = pd.DataFrame(X_test)
    test_data[len(test_data.columns)] = y_test
    test_data.to_csv(test_file, index=False, header=False)
    paths_files = Path(__file__).parent.parent / "data" / "susy_paths.txt"
    with open(paths_files, 'w') as f:
        f.write(f"{train_file}\n{test_file}\n")

    return str(train_file), str(test_file)

if __name__ == "__main__":
    try:
        preprocess_data()
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        exit(1)
    print("Data preprocessing complete. Train and test data saved to CSV files.")