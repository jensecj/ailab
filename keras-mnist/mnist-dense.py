# Resources:
# http://ankivil.com/mnist-database-and-simple-classification-networks/
# http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
# http://cs231n.github.io/neural-networks-3/#ada

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils

(training_features, training_labels), (testing_features, testing_labels) = mnist.load_data()

print(f"training_features shape: {training_features.shape}")
print(f"testing_features shape: {testing_features.shape}")

print(f"training_labels shape: { training_labels.shape}")
print(f"testing_labels shape: {testing_labels.shape}")

#################
# preprocessing #
#################

# flatten 2D image data into a 1D vector, 784 = 28*28
training_features = training_features.reshape(60000, 784)
testing_features = testing_features.reshape(10000, 784)

# normalize pixel values to be floats in [0;1] instead of integers in [0;255]
training_features = training_features.astype('float32')
testing_features = testing_features.astype('float32')

training_features /= 255.
testing_features /= 255.

print(training_features.shape[0], 'training samples')
print(testing_features.shape[0], 'testing samples')

num_features = training_features.shape[1]
num_classes = np.unique(testing_labels).size
print(f"Training set has {num_features} features and {num_classes} classes.")

# convert class vectors to binary class matrices (i.e. one-hot vectors)
training_labels = np_utils.to_categorical(training_labels, num_classes)
testing_labels = np_utils.to_categorical(testing_labels, num_classes)

######################
# model architecture #
######################

# define the model
model = Sequential([
    Dense(512, input_shape=(num_features,)),
    Activation('relu'),

    Dropout(0.5),

    Dense(512),
    Activation('relu'),

    Dropout(0.5),

    Dense(num_classes),
    Activation('softmax')
])

# compile our model we're using categorical crossentropy because we are dealing
# with categories (0-9), and crossentropy is a fine loss-function for this case.
opt = Adam()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

############
# training #
############

batch_size = 128 # number of examples used in each optimization step
epochs = 20 # number of times the whole data is used to learn

# train our model on the data
history = model.fit(training_features, training_labels,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = (testing_features, testing_labels))

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
score = model.evaluate(testing_features, testing_labels, verbose=0)

print(f'Test score: {score[0]}')
print(f'Test accuracy: {score[1] * 100:.2f}%')

# around 98%, not bad.
