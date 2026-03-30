"""Example: train a neural network on the Wine housing data.

"""

# import custom deep learning library
from deepnum.data.data import train_test_val_split, DataLoader
from deepnum.data.transformations import OneHotEncoder, NormaliseData
from deepnum.data.loader import internet_loader
from deepnum.constructor import sequential_model
from deepnum.layers.linear import layer_linear
from deepnum.layers.activation import af_relu, af_softmax
from deepnum.layers.regularisation import reg_dropout
from deepnum.loss import loss_cross_entropy
from deepnum.optimiser import optimiser_sgd
  
# transformations
ohe = OneHotEncoder()
normaliser = NormaliseData()
    
# load data
split_percent=[0.7, 0.15, 0.15]
label_feature = 0 # index
excludeList = []

data = internet_loader("wine", shuffle=True) # wine dataset
y = data[:, label_feature]
y = ohe.fit_transform(y) # one hot encoding
X = data[:, [i for i in range(data.shape[1]) if i not in excludeList + [label_feature]]]

X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(X, y, split_percent=split_percent)

# normalise data
X_train = normaliser.fit_transform(X_train)
X_val = normaliser.transform(X_val)
X_test = normaliser.transform(X_test)

# setup network
layers = [
    layer_linear(X_train.shape[1], 20),
    af_relu(),
    layer_linear(20, 3),
    # af_softmax() # moved inside cross entropy
]
model = sequential_model(layers=layers)



# # training setup
# num_epochs = 200
# batch_size = 16
# loss = None
# lr = 0.00001
# optimiser = None

# # load data
# train_dataloader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(X_val, y_val)
# test_dataloader = DataLoader(X_val, y_val)