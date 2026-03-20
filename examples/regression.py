"""Example: train a neural network on Boston housing data.

Demonstrates loading data, building a model with linear layers, ReLU activation and dropout, then training with SGD.
"""

# import custom deep learning library
from deepnum.data.data import train_test_val_split, NormaliseData, DataLoader
from deepnum.data.loader import internet_loader
from deepnum.constructor import sequential_model
from deepnum.layers.linear import layer_linear
from deepnum.layers.activation import af_relu
from deepnum.layers.regularisation import reg_dropout
from deepnum.loss import loss_mse
from deepnum.optimiser import optimiser_sgd

# load data
split_percent=[0.7, 0.15, 0.15]
label_feature = 13 # index
excludeList = [0, 1, 2, 3, 6, 8, 9, 11]  # keep: RM(5), NOX(4), DIS(7), PTRATIO(10), LSTAT(12)

data = internet_loader("boston")
X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(data=data, label_feature=label_feature, split_percent=split_percent, excludeList=excludeList)

# normalise data
normaliser = NormaliseData()
X_train = normaliser.fit_transform(X_train)
X_val = normaliser.transform(X_val)
X_test = normaliser.transform(X_test)

# setup network
layers = [
    layer_linear(X_train.shape[1], 20),
    af_relu(),
    reg_dropout(0.1),
    layer_linear(20, 20),
    af_relu(),
    reg_dropout(0.1),
    layer_linear(20, 1),
]
model = sequential_model(layers=layers)

# training setup
num_epochs = 200
batch_size = 16
loss = loss_mse()
lr = 0.00001
optimiser = optimiser_sgd(loss=loss, learning_rate=lr)

# load data
train_dataloader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(X_val, y_val)
test_dataloader = DataLoader(X_val, y_val)

# Training iteration
for i in range(num_epochs):  
    model.train()
    train_loss_sum = 0.0 # sum of batch losses weighted by batch size
    train_sample_count = 0 # total samples seen this epoch
    
    for X_train, y_train in train_dataloader:
        # forward to get loss
        train_pred = model.forward_pass(X_train)
        train_batch_loss = loss.loss_value(train_pred, y_train)
        
        B_train = X_train.shape[0] # batch size
        train_loss_sum += train_batch_loss * B_train
        train_sample_count += B_train

        # backward to get gradients and update weights
        optimiser.backward_pass(model, train_pred, y_train)

        model.zero_grad() # zero grad to reset for next batch
    
    # get validation predictions each epoch
    val_loss_sum = 0.0
    val_sample_count = 0 
    model.eval()
    for X_val, y_val in val_dataloader:
        val_pred = model.forward_pass(X_val) 
        v_loss = loss.loss_value(val_pred, y_val)

        val_size = X_val.shape[0]
        val_loss_sum += v_loss * val_size
        val_sample_count += val_size

    train_loss_avg = train_loss_sum / max(1, train_sample_count)
    val_loss_avg = val_loss_sum / max(1, val_sample_count)

    print(f"Epoch: {i+1} | Training Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

test_loss_sum = 0.0
test_sample_count = 0
model.eval() 
for X_test, y_test in test_dataloader:
    test_pred = model.forward_pass(X_test) 
    t_loss = loss.loss_value(test_pred, y_test)

    test_size = X_test.shape[0]
    test_loss_sum += t_loss * test_size
    test_sample_count += test_size

test_loss_avg = test_loss_sum / max(1, test_sample_count)
print(f"\nFinal Test Loss: {test_loss_avg:.4f}")