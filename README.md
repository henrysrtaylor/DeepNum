# DeepNum
A modular Deep Learning framework built from scratch using NumPy. Implements different layers, activation functions, and loss with backpropagation. Developed for self-educational purposes to bridge the gap between theory and implementation and test my own low level understanding.

Designed with a modularity inspired by PyTorch and scikit-learn, the API will feel familiar to users of those libraries. For example, the data normalization class follows the scikit-learn fit/transform pattern, while model architecture can be constructed using a Sequential API similar to PyTorch.

> NOTE: AI was specifically not used to generate any of the core code in this repo as it was for self-educational purposes.

## 📋 Core Components
- Layers: Linear
- Optimisation: SGD
- Loss: Mean Squared Error, Cross Entropy
- Activations: ReLu (+ Leaky), Softmax, Sigmoid
- Backpropagation: Implementation via the chain-rule
- Regularisation: Dropout layer
- Data: CSV parser, internet data loading, train-test split, data normalisation, dataloader
- Metrics: Accuracy
- Examples: Regression (Boston Housing)

## 📈 Future Roadmap
Additions planned:
- Metrics: Precision, Recall, F1, MSE, MAE, RMSE
- Optimisation: Adam
- Examples: Image Classification (MNIST)
- Layers: CNN, RNN, LSTM
- Regularisation: batch norm, spatial dropout layer for CNN
- Training: Cosine decay

## 📂 Project Structure
```
DeepNum/
├── deepnum/           # Core library
│   ├── data/          # Data loading & preprocessing
│   ├── layers/        # Layer definitions
│   ├── constructor.py # Model building
│   ├── loss.py        # Loss functions
│   └── optimiser.py   # Optimisers
├── examples/          # Examples using DeepNum package
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
