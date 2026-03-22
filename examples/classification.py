"""Example: train a neural network on the Wine housing data.

"""

# import custom deep learning library
from deepnum.data.data import train_test_val_split, DataLoader
from deepnum.data.loader import internet_loader
from deepnum.constructor import sequential_model


# load data
split_percent=[0.7, 0.15, 0.15]
label_feature = 0 # index
excludeList = []

data = internet_loader("wine", shuffle=True) # wine dataset
X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(data=data, label_feature=label_feature, split_percent=split_percent, excludeList=excludeList)

print(y_train)

# One hot encoding labels
# FIT > DICT OF ALL UNIQUE LABELS TO POS.. then transform and fit transform to transform them > last function to go from one hot to label








# # setup network
# layers = [

# ]
# model = sequential_model(layers=layers)

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