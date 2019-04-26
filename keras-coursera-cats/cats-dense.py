import os

import h5py
from PIL import Image
import scipy
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils

#############
# load data #
#############

train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
train_features = np.array(train_dataset["train_set_x"][:]) # training set features
train_labels = np.array(train_dataset["train_set_y"][:]) # training set labels

test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
test_features = np.array(test_dataset["test_set_x"][:]) # test set features
test_labels = np.array(test_dataset["test_set_y"][:]) # test set labels

descriptions = np.array(test_dataset["list_classes"][:]) # list of classes

print(f"train_features shape: {train_features.shape}")
print(f"test_features shape: {test_features.shape}")

print(f"train_labels shape: {train_labels.shape}")
print(f"test_labels shape: {test_labels.shape}")

#################
# preprocessing #
#################

# flatten 2D image data into a 1D vector
train_features = train_features.reshape(train_features.shape[0], 64*64*3)
test_features = test_features.reshape(test_features.shape[0], 64*64*3)

# normalize pixel values to be floats in [0;1] instead of integers in [0;255]
train_features = train_features.astype('float32')
test_features = test_features.astype('float32')

train_features /= 255.
test_features /= 255.

print(f"flattened training features: {train_features.shape}")
print(f"flattened testing features: {test_features.shape}")

print(train_features.shape[0], 'training samples')
print(test_features.shape[0], 'testing samples')

num_features = train_features.shape[1]
num_classes = np.unique(test_labels).size
print(f"Training set has {num_features} features and {num_classes} classes.")

######################
# model architecture #
######################

# define the model
model = Sequential([
    Dense(512, input_shape=(num_features,)), Activation('relu'),

    Dropout(0.2),

    Dense(256), Activation('relu'),

    Dropout(0.3),

    Dense(256), Activation('relu'),

    Dropout(0.2),

    Dense(1), Activation('sigmoid')
])

# compile our model we're using categorical crossentropy because we are dealing
# with categories (0-9), and crossentropy is a fine loss-function for this case.
opt = Adam(lr = 0.0001, decay = 1e-7)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

############
# training #
############

batch_size = 64 # number of examples used in each optimization step
epochs = 75 # number of times the whole data is used to learn

# train our model on the data
history = model.fit(train_features, train_labels,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = (test_features, test_labels))

model.save("dense-model.h5")
# model = load_model("dense-model.h5")

##############
# evaluation #
##############

# plot how the model did while training

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# evaluate how the model does on the test set
score = model.evaluate(test_features, test_labels, verbose=0)

print(f'Test score: {score[0]}')
print(f'Test accuracy: {score[1] * 100:.2f}%')

for i in np.random.choice(test_features.shape[0], 5):
    example = np.array([test_features[i]], dtype='float')

    pixels = example.reshape((64,64,3))
    plt.imshow(pixels)
    plt.show()

    prediction_idx = model.predict(example).squeeze()
    print(prediction_idx)
    prediction_idx = int(np.round(prediction_idx))
    print(prediction_idx)
    prediction = descriptions[prediction_idx]
    print(f"The model predicted {prediction_idx} = '{prediction}'")
