# DeepNum
A modular Deep Learning framework built from scratch using NumPy. Implements different layers, activation functions, and loss with backpropagation. Developed for self-educational purposes to bridge the gap between high-level theory and low-level implementation.

> NOTE: AI was specifically not used to generate any of the core code in this repo as it was for self-educational purposes.
> NOTE: You may notice many of the classes and modules are inspired by trying to recreate the feel/interface PyTorch and Sklearn from my own usage.

## 📈 Future Roadmap
Additions planned:
- Regularization: dropout, batch norm
- Optimisation: Adam
- Layers: CNN, RNN, LSTM

## 📋 Core Components
- Layers: Linear
- Optimisation: SGD
- Loss: Mean Squared Error
- Backpropagation: Implementation via the chain-rule
- Data: CSV parser, internet data loading, train-test split, data normalisation, dataloader 
- Examples: Regression (Boston Housing)

## 📂 Project Structure
```
DeepNum/
├── deepnum/           # Core library
│   ├── data/          # Data loading & preprocessing
│   └── model/         # Architecture & Layer definitions
├── examples/          # Demos & tutorials
├── tests/             # Mathematical verification
└── pyproject.toml     # Package configuration
```

## 🚀 Getting Started

### 1. Clone the Repository
Open your terminal and run:
```bash
git clone "https://github.com/henrysrtaylor/DeepNum.git"
cd DeepNum
```

### 2. Set Up Virtual Environment
Windows:
```
python -m venv venv
.\venv\Scripts\activate
```

mac/linux:
```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Library
Make sure you are in project root then:
```
pip install -e .
```