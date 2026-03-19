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

def internet_loader(dataset_name:str):
    if not isinstance(dataset_name, str):
        raise ValueError("dataset_name must be a string")
    
    match dataset_name.lower():
        case "boston":
            return _load_boston_from_url()
        case _:
            raise NotImplementedError(f"dataset_name '{dataset_name}' is not supported")
        
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