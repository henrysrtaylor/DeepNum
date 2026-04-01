
"""Data transformation utilities.

Provides NormaliseData for standardisation (zero mean, unit variance).
"""

import numpy as np

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

class OneHotEncoder():
    """Provide list to encode features. If no list then assumes label encoder."""
    def __init__(self, index:list=None):
        if index is not None and not isinstance(index, list):
            raise ValueError("index must be list or None.")
        
        self.index = index
    
    def _validation(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array.")
        
    def fit_transform(self, X:np.ndarray) -> np.ndarray:
        self._validation(X)
        
        if self.index:  # features (2D)
            if X.ndim != 2:
                raise ValueError("X must be 2D for feature encoding.")
            parts = []
            for col in range(X.shape[1]):
                if col in self.index:
                    unique = np.unique(X[:, col])
                    one_hot = (X[:, col:col+1] == unique).astype(float)
                    parts.append(one_hot)
                else:
                    parts.append(X[:, col:col+1].astype(float))
            return np.hstack(parts)
        
        else:  # labels (1D)
            if X.ndim != 1:
                raise ValueError("X must be 1D for label encoding.")
            classes = np.unique(X)
            one_hot = (X[:, None] == classes).astype(float)
            return one_hot   