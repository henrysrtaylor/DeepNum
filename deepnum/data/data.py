"""Data splitting and batching utilities.

Provides train_test_val_split for partitioning X/y arrays,
and DataLoader for batching during training.
"""

import numpy as np

def train_test_val_split(X:np.ndarray, y:np.ndarray, split_percent:list=[0.7, 0.15, 0.15]) -> tuple:
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array.")
    
    if len(X) != len(y):
        raise ValueError(f"Length mismatch: X has {len(X)} rows, but y has {len(y)}.")
                
    if not isinstance(split_percent, list) or len(split_percent) != 3:
        raise ValueError("split_percent must be a list of 3 floats (Train, Test, Val).")
    
    if not np.isclose(sum(split_percent), 1.0):
        raise ValueError(f"Split percentages must sum to 1.0. Currently: {sum(split_percent)}")
    
    num_samples = len(X)
    split_indices = (np.cumsum(split_percent) * num_samples).astype(int)
    
    X_train, y_train = X[:split_indices[0]], y[:split_indices[0]]
    X_test, y_test = X[split_indices[0]:split_indices[1]], y[split_indices[0]:split_indices[1]]
    X_val, y_val = X[split_indices[1]:split_indices[2]], y[split_indices[1]:split_indices[2]]
    
    return X_train, y_train, X_test, y_test, X_val, y_val          

class DataLoader:
    """
    Takes a dataset and yields batches of data for training based on X_train and y_train inputs.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size:int=8, shuffle:bool = False):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array.")
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an int.")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean.")
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X has {len(X)} rows, but y has {len(y)}.")

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        num_samples = self.X.shape[0]
        indices = np.arange(num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, num_samples, self.batch_size):
            batch_idx = indices[start_idx : start_idx + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]