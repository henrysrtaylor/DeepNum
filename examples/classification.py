"""Example: train a neural network on Wine dataset for multi-class classification.

Demonstrates loading data, one-hot encoding labels, building a model with linear
layers, ReLU activation and dropout, then training with SGD and cross entropy loss.
"""

import numpy as np

from deepnum.constructor import SequentialModel

# import custom deep learning library
from deepnum.data.data import DataLoader, train_test_val_split
from deepnum.data.loader import internet_loader
from deepnum.data.transformations import NormaliseData, OneHotEncoder
from deepnum.layers.activation import AfRelu
from deepnum.layers.linear import LayerLinear
from deepnum.layers.regularisation import RegDropout
from deepnum.loss import LossCrossEntropy
from deepnum.metrics import metric_accuracy
from deepnum.optimiser import OptimiserBase

# transformations
ohe = OneHotEncoder()
normaliser = NormaliseData()

# load data
split_percent = [0.7, 0.15, 0.15]
label_feature = 0  # index
excludeList = []

data = internet_loader("wine", shuffle=True)  # wine dataset
y = data[:, label_feature]
y = ohe.fit_transform(y)  # one hot encoding
X = data[:, [i for i in range(data.shape[1]) if i not in excludeList + [label_feature]]]

X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(
    X, y, split_percent=split_percent
)

# normalise data
X_train = normaliser.fit_transform(X_train)
X_val = normaliser.transform(X_val)
X_test = normaliser.transform(X_test)

# setup network
layers = [
    LayerLinear(X_train.shape[1], 20),
    AfRelu(),
    RegDropout(0.1),
    LayerLinear(20, 20),
    AfRelu(),
    RegDropout(0.1),
    LayerLinear(20, 3),
    # af_softmax() # moved inside cross entropy
]
model = SequentialModel(layers=layers)

# training setup
num_epochs = 220
batch_size = 8
loss = LossCrossEntropy()
lr = 0.001
optimiser = OptimiserBase(loss=loss, learning_rate=lr)

# load data
train_dataloader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(X_val, y_val)
test_dataloader = DataLoader(X_val, y_val)

# Training iteration
for i in range(num_epochs):
    model.train()
    train_loss_sum = 0.0  # sum of batch losses weighted by batch size
    train_sample_count = 0  # total samples seen this epoch
    train_preds_all, train_labels_all = [], []

    for X_train, y_train in train_dataloader:
        # forward to get loss
        train_pred = model.forward_pass(X_train)
        train_batch_loss = loss.loss_value(train_pred, y_train)

        B_train = X_train.shape[0]  # batch size
        train_loss_sum += train_batch_loss * B_train
        train_sample_count += B_train

        # append predictions for accuracy calculation
        train_preds_all.append(train_pred)
        train_labels_all.append(y_train)

        # backward to get gradients and update weights
        optimiser.backward_pass(model, train_pred, y_train)

        model.zero_grad()  # zero grad to reset for next batch

    # get validation predictions each epoch
    val_loss_sum = 0.0
    val_sample_count = 0
    val_preds_all, val_labels_all = [], []

    model.eval()
    for X_val, y_val in val_dataloader:
        val_pred = model.forward_pass(X_val)
        v_loss = loss.loss_value(val_pred, y_val)

        val_size = X_val.shape[0]
        val_loss_sum += v_loss * val_size
        val_sample_count += val_size

        # append predictions for accuracy calculation
        val_preds_all.append(val_pred)
        val_labels_all.append(y_val)

    train_loss_avg = train_loss_sum / max(1, train_sample_count)
    val_loss_avg = val_loss_sum / max(1, val_sample_count)

    # compute accuracy
    train_acc = metric_accuracy(np.vstack(train_preds_all), np.vstack(train_labels_all))
    val_acc = metric_accuracy(np.vstack(val_preds_all), np.vstack(val_labels_all))

    print(
        f"Epoch: {i + 1} | Train Loss: {train_loss_avg:.4f}"
        f" | Train Acc: {train_acc:.4f} | Val Loss: {val_loss_avg:.4f}"
        f" | Val Acc: {val_acc:.4f}"
    )

test_loss_sum = 0.0
test_sample_count = 0
test_preds_all, test_labels_all = [], []

model.eval()
for X_test, y_test in test_dataloader:
    test_pred = model.forward_pass(X_test)
    t_loss = loss.loss_value(test_pred, y_test)

    test_size = X_test.shape[0]
    test_loss_sum += t_loss * test_size
    test_sample_count += test_size

    test_preds_all.append(test_pred)
    test_labels_all.append(y_test)

test_loss_avg = test_loss_sum / max(1, test_sample_count)
test_acc = metric_accuracy(np.vstack(test_preds_all), np.vstack(test_labels_all))
print(f"\nFinal Test Loss: {test_loss_avg:.4f} | Test Acc: {test_acc:.4f}")
