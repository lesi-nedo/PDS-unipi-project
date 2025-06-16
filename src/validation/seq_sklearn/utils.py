import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_dataset(file_path="../../data/iris/iris.data"):
    """
    Load dataset from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    tuple: Features (X), labels (y), and label encoder.
    If the file cannot be read, returns None for all.
    
    """
    try:
        data = pd.read_csv(file_path, header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        return X, y, label_encoder
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {file_path}")
    except pd.errors.ParserError:
        raise ValueError(f"Error parsing file: {file_path}")