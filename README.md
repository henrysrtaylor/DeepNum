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
- Data: CSV parser, internet data loading, train-test split, data normalisation, dataloader, one-hot encoding
- Metrics: Accuracy
- Examples: Regression (Boston Housing), Classification (Wine)

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
│   ├── metrics.py     # Evaluation metrics
│   └── optimiser.py   # Optimisers
├── examples/          # Examples using DeepNum package
└── pyproject.toml     # Package configuration
```

## 🚀 Getting Started

### 1. Install Requirements

You need Python 3.10 or later, Git, and [uv](https://docs.astral.sh/uv/).

Install uv on Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install uv on Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/henrysrtaylor/DeepNum.git
cd DeepNum
```

### 3. Install Dependencies

From the project root, create or update the local virtual environment and install DeepNum and its dependencies:

```bash
uv sync
```

Activate the virtual environment if you want to run commands directly inside it.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Alternatively, prefix commands with `uv run` without activating the environment.
