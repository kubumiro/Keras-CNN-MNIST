import numpy as np
import numpy.random
import matplotlib.pyplot as plt

import tensorflow
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, Convolution2D, Flatten, merge
from keras.utils import np_utils
from keras.optimizers import SGD, Adam



def data_generator(X, Y, batchsize):
    nb_classes = 10
    N = X.shape[0]
    while True:
        indices1 = np.random.randint(low=0, high=N, size=(batchsize,))  # randomly draw a set of sample indices
        indices2 = np.random.randint(low=0, high=N, size=(batchsize,))

        X1 = X[indices1, ...].astype('float32') / 255.0
        X2 = X[indices2, ...].astype('float32') / 255.0
        Y1 = Y[indices1]
        Y2 = Y[indices2]


        X1 = np.expand_dims(X1, axis=1)  # For conv with theano, shape=(batchsize, channels, row, col).
        X2 = np.expand_dims(X2, axis=1)  # We are just adding a dummy dimension saying that there is one channel.

        Y1 = np_utils.to_categorical(Y1, nb_classes)
        Y2 = np_utils.to_categorical(Y2, nb_classes)


        yield {'input1': X1, 'input2': X2}, {'aux1': Y1, 'aux2': Y2}


# Load data.
(X_train, Y_train), (X_valid, Y_valid) = mnist.load_data()  # Shape = (N,28,28), (N,)

# Create generators.
batchsize = 200
data_train = data_generator(X_train, Y_train, batchsize)
data_valid = data_generator(X_valid, Y_valid, batchsize)






input1 = Input(shape=(1, 28, 28), dtype='float32', name='input1')  # Argument 'name' must match name in dictionary.

nb_filter = 32  # Number of convolutional kernels.
nb_row, nb_col = 7, 7  # Convolution kernel size.
subsample = (3, 3)  # Step size for convolution kernels.

conv = Convolution2D(nb_filter, (nb_row, nb_col), activation='relu', padding='same', strides=subsample)  # shared layer
x1 = conv(input1)  # Layer object conv transforms data.

x1 = Flatten()(x1)


layer = Dense(256, activation='relu')
x1 = layer(x1)

layer = Dense(10)  # These weights are shared.
aux1 = Activation(activation='softmax', name='aux1')(layer(x1))  # Output layers must be named.

model = Model(inputs=[input1], outputs=[aux1])
model.summary()

optimizer = Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-08) # Optimization hyperparameters.

model.compile(optimizer=optimizer,
              loss={'aux1':'categorical_crossentropy'},
#               loss_weights={'out': 1.0, 'aux1': 1.0, 'aux2':1.0}, # These can be tuned.
              loss_weights={'aux1': 1.0}, # These can be tuned.
              metrics=['accuracy'])


import os
import time
from keras.callbacks import TensorBoard
log_dir = './tb_log/' + time.strftime("%c")
log_dir = log_dir.replace(' ', '_').replace(':', '-')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tb = TensorBoard(log_dir=log_dir,
                 histogram_freq=0,
                 write_graph=False,
                 write_grads=False,
                 write_images=True)

callbacks = [tb]#[stopping]

history   = model.fit_generator(generator=data_train, steps_per_epoch=10,
                              epochs=10, verbose=1,
                              callbacks=callbacks,
                              validation_data=data_valid, validation_steps=10)


# Plot loss trajectory throughout training.
plt.figure(1, figsize=(10,5))
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.xlim((0, 10))
plt.ylim((0, 1))
plt.show()


import keras.backend as K





def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):

    print('----- activations -----')

    activations = []

    inp = model.input



    model_multi_inputs_cond = True

    if not isinstance(inp, list):

        # only one input! let's wrap it in a list.

        inp = [inp]

        model_multi_inputs_cond = False



    outputs = [layer.output for layer in model.layers if

               layer.name == layer_name or layer_name is None]  # all layer outputs



    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions



    if model_multi_inputs_cond:

        list_inputs = []

        list_inputs.extend(model_inputs)

        list_inputs.append(0.)

    else:

        list_inputs = [model_inputs, 0.]



    # Learning phase. 0 = Test mode (no dropout or batch normalization)

    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]

    layer_outputs = [func(list_inputs)[0] for func in funcs]

    for layer_activations in layer_outputs:

        activations.append(layer_activations)

        if print_shape_only:

            print(layer_activations.shape)

        else:

            print(layer_activations)

    return activations





def display_activations(activation_maps):

    import numpy as np

    import matplotlib.pyplot as plt

    """

    (1, 26, 26, 32)

    (1, 24, 24, 64)

    (1, 12, 12, 64)

    (1, 12, 12, 64)

    (1, 9216)

    (1, 128)

    (1, 128)

    (1, 10)

    """

    batch_size = activation_maps[0].shape[0]

    assert batch_size == 1, 'One image at a time to visualize.'

    for i, activation_map in enumerate(activation_maps):

        print('Displaying activation map {}'.format(i))

        shape = activation_map.shape

        if len(shape) == 4:

            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))

        elif len(shape) == 2:

            # try to make it square as much as possible. we can skip some activations.

            activations = activation_map[0]

            num_activations = len(activations)

            if num_activations > 1024:  # too hard to display it on the screen.

                square_param = int(np.floor(np.sqrt(num_activations)))

                activations = activations[0: square_param * square_param]

                activations = np.reshape(activations, (square_param, square_param))

            else:

                activations = np.expand_dims(activations, axis=0)

        else:

            raise Exception('len(shape) = 3 has not been implemented.')

        plt.imshow(activations, interpolation='None', cmap='gray')
        plt.show()

inp_a = np.random.uniform(size=(100, 10))
data=X_train[0]
plt.imshow(data.reshape(28,28), cmap='gray', interpolation='nearest')
print(data.shape)
# reshape
data = data.reshape((1, data.shape[0], data.shape[1]))
print(data.shape)
data = data.reshape((1, 1, 28, 28))
print(data.shape)

print(type([inp_a[0]]))
print(type(X_train[0]))
print(type(list(X_train[0])))
print(len(list(X_train[0])))
print([inp_a[0],inp_a[0]])
print(len([inp_a[0]]))
print(len([inp_a[0],inp_a[0]]))
a= get_activations(model,list(data), print_shape_only=True)
display_activations(a)
plt.show()

