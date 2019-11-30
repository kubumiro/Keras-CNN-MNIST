
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.regularizers import l2

# dashboard
import os
import time
from keras.callbacks import TensorBoard
# training variables
batch_size = 300
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # use Flatten() to turn shape from (num_samples, rows, columns, channels) to (num_samples, dimensions)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax')) # output has 10 dimensions

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

log_dir = './tb_log/' + time.strftime("%c")
log_dir = log_dir.replace(' ', '_').replace(':', '-')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tb = TensorBoard(log_dir=log_dir,
                 histogram_freq=0,
                 write_graph=False,
                 write_grads=False,
                 write_images=False)
history = model.fit(x_train, y_train, batch_size=300, epochs=1, verbose=1,
                    validation_data=(x_test, y_test), callbacks=[tb])

plt.figure(1, figsize=(10,5))
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()

plt.xlim((-1, 10))
plt.ylim((0, 1))
plt.show()

# Plot the weights of one specific layer in your model

from kerastoolbox.visu import plot_all_feature_maps

images = X_test[:4]
_ = plot_all_feature_maps(model, X=images, n_columns=2, n=256)