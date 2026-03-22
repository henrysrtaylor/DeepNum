"""Functions for loading datasets from files or the internet.

Provides csv_loader for local files and internet_loader for fetching common datasets like Boston housing.
"""

import numpy as np
import os
import requests

def csv_loader(file_path:str, delimiter:str=',', skip_header:int=0, encoding:str='utf-8', dtype=float) -> np.ndarray:
    if not (isinstance(file_path, str) and file_path.lower().endswith((".txt", ".csv"))):
        raise ValueError("file_path must be a string ending in .txt or .csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file at: {file_path}")
    
    data = np.genfromtxt(file_path, delimiter=delimiter, skip_header=skip_header, dtype=dtype, encoding=encoding)
    return data

def internet_loader(dataset_name:str, shuffle:bool = False):
    if not isinstance(dataset_name, str):
        raise ValueError("dataset_name must be a string")
    
    match dataset_name.lower():
        case "boston":
            data = _load_boston_from_url()
        case "wine":
            data = _load_wine_from_url()
        case _:
            raise NotImplementedError(f"dataset_name '{dataset_name}' is not supported")
    
    if shuffle:
        np.random.shuffle(data)

    return data
        
def _load_boston_from_url():
    url = "http://lib.stat.cmu.edu/datasets/boston"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch data. Status: {response.status_code}")

    # The data starts after row 22
    lines = response.text.splitlines()[22:]
    
    raw_values = []
    for i in range(0, len(lines), 2):
        # Line 1 contains first 11 columns, Line 2 contains last 3 (including target)
        combined_row = lines[i].split() + lines[i+1].split()
        raw_values.append([float(x) for x in combined_row])
    
    data = np.array(raw_values)

    return data

def _load_wine_from_url():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch data. Status: {response.status_code}")

    # The wine dataset is a standard CSV with no header
    lines = response.text.strip().splitlines()
    
    raw_values = []
    for line in lines:
        if line:
            # Each row is: class, feat1, feat2, ..., feat13
            row = [float(x) for x in line.split(',')]
            raw_values.append(row)
    
    data = np.array(raw_values)

    # Column 0 is the Label (y), Columns 1-13 are the Features (X)
    # shift from 1,2,3 to 0,1,2
    data[:, 0] = data[:, 0] - 1

    return data