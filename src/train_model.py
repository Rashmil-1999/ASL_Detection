import keras
import os
import cv2
import numpy
from itertools import repeat
import importlib
from cnn_model import buildCNNModel
import prepare_data

# ===define few parameters===

# general parameters for project
num_classes = 29
no_of_epochs = 10
size = [64, 64]
batch_size = 32

# network parameters define
kernel_size = [(7, 7), (5, 5), (3, 3)]
filters = [16, 32, 64]
pool_size = (2, 2)
dropout = 0.05
stride = 1

# compile parameters define
loss = "categorical_crossentropy"
optimizer = "adam"
metrics = ["accuracy"]

# define paths
models_path = "C:\\programming\\Quick-CNN-training\\trained_model\\"
pathToDatasetTrain = "C:\\programming\\Quick-CNN-training\\dataset\\asl-alphabet\\asl_alphabet_train\\"
# pathToDatasetTest = "../dataset/asl-alphabet/asl_alphabet_test/"

# use the prprocessing of images and then prepare the data and load it into inputs and outputs
inputs, outputs = prepare_data.prepareData(size, pathToDatasetTrain, num_classes)
inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], inputs.shape[2], 1)

num_of_samples = len(inputs)
# divide into 80-20 train-test set
train_data_length = int(num_of_samples * 0.8)

x_train, x_test = inputs[0:train_data_length], inputs[train_data_length:]
y_train, y_test = outputs[0:train_data_length], outputs[train_data_length:]

model = buildCNNModel(
    inputs.shape[1:],
    num_classes,
    filters=filters,
    kernel_size=kernel_size,
    pool_size=pool_size,
    dropout=dropout,
    stride=stride,
)
# model = buildCnnModel(inputs.shape[1:], num_classes, 32, (3,3), 0.05, (2,2), 1)

print(model.summary())
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=no_of_epochs,
    validation_data=(x_test, y_test),
)

if not os.path.exists(models_path):
    os.makedirs(models_path)
model.save(
    models_path
    + "asl-sign_recod_%d_acc-%f.model" % (no_of_epochs, history.history["val_acc"][4])
)

