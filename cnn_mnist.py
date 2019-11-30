import os


#os.environ['KERAS_BACKEND'] = 'tensorflow'

import matplotlib.pyplot as plt
import numpy as np

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

# Load dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Only take a small part of the data to reduce computation time

X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:5000]
y_test = y_test[:5000]

# Define some variables from the dataset

nb_classes = 10
img_rows, img_cols = X_train.shape[-2:]

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols).astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Model parameters

nb_filters = 32
nb_pool = 2
kernel_size = (3, 3)

# Create the model

model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1), padding="valid"))


#model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
#                        border_mode='valid',activation="relu"))

#model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

batch_size = 128
nb_epoch = 5

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
