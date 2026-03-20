"""Data splitting, normalisation, and batching utilities.

Provides train_test_val_split for partitioning data,
NormaliseData for standardisation, and DataLoader for batching.
"""

import numpy as np

def train_test_val_split(data:np.ndarray, label_feature:int, split_percent:list=[0.7, 0.15, 0.15], excludeList:list=[]) -> tuple:
    if not isinstance(data, (np.ndarray, list)):
        raise TypeError("Data must be a numpy array or a list.")

    if not isinstance(label_feature, int):
        raise TypeError("label_feature must be an integer ")
                
    if not isinstance(split_percent, list) or len(split_percent) != 3:
        raise ValueError("split_percent must be a list of 3 floats (Train, Test, Val).")
    
    if not np.isclose(sum(split_percent), 1.0):
        raise ValueError(f"Split percentages must sum to 1.0. Currently: {sum(split_percent)}")
    
    if not isinstance(excludeList, list):
        raise TypeError("excludeList must be a list of integers.")
    
    data = np.array(data)
    num_feature_cols = data.shape[1]
    num_feature_rows = data.shape[0]
    
    y = data[:, label_feature].reshape(-1,1)
    X = data[:, [i for i in range(num_feature_cols) if i not in excludeList+[label_feature]]]

    train_test_val_index_split = (np.cumsum(split_percent) * num_feature_rows).astype(int)
    
    X_train, y_train = X[0:train_test_val_index_split[0]], y[0:train_test_val_index_split[0]]
    X_test, y_test = X[train_test_val_index_split[0]:train_test_val_index_split[1]], y[train_test_val_index_split[0]:train_test_val_index_split[1]]
    X_val, y_val = X[train_test_val_index_split[1]:train_test_val_index_split[2]], y[train_test_val_index_split[1]:train_test_val_index_split[2]]
    
    return X_train, y_train, X_test, y_test, X_val, y_val

class NormaliseData():
    def __init__(self):
        self.X_mean = None
        self.X_std = None
    
    def _validation(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array.")
    
    def _check_params(self):
        if self.X_mean is None or self.X_std is None:
            raise AttributeError("Call fit on train set first.")

    def fit(self, X):
        self._validation(X)
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-8 # add epsilon to avoid divising by zero

    def transform(self, X):
        self._validation(X)
        self._check_params()

        return (X - self.X_mean) / self.X_std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

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