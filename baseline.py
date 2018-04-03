#!python

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, LSTM
from tensorflow import keras
from helper_funct import *
import time


# HYPER Parameters
mini_batch_size = 32
embedding_size = 8
learning_rate = 0.001
epoch_train = 1#300  # maximum repetitions
validation_split = 0.05
optimizer = RMSprop(lr=learning_rate)
metrics = ['accuracy']
bias_init = 'random_normal'

# Model parameters
units = 500
layers = 1
lstm_type = 'LSTM'
activation = 'softmax'
loss_function = 'binary_crossentropy'

# Data specific parameters
n_sets = 8
n_folds = 4
num_classes = 2
time_steps = 1

# Defined callbacks. One for tensorboard and another for stopping the training at loss < 0.0005
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
EarlyStopping = EarlyStoppingByLossVal(monitor='loss', value=0.0005, verbose=1)
callbacks = [EarlyStopping]

# initialing the output array
bal_accuracy = np.zeros(n_folds)

# Reading the data from the read_data function
x_train, x_test, y_train, y_test = read_data(n_folds)

# in_shape defines the input shape of the LSTM modules
in_shape = len(x_train[0][0][0])  # data length variable for the input tensor

# Defining the tensorflow/Keras model
model = Sequential()
# Adding the LSTM layers or the Bidirectional LSTM modules
model.add(LSTM(units=units, input_shape=(time_steps, in_shape), bias_initializer=bias_init))
# Adding the rest of the network's components
model.add(Dense(units=num_classes, bias_initializer=bias_init))
model.add(Activation(activation=activation))
model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
model.summary()

start_time = time.time()

# training of the model
bal_accuracy = train_model(x_train, y_train, validation_split, epoch_train, mini_batch_size,
                           callbacks, x_test, y_test, model, n_folds, learning_rate)

# Saving the balanced accuracy over the 4 folds
balanced_accuracy = mean(bal_accuracy[:])
run_time = time.time() - start_time
print "bal_accuracy: ", bal_accuracy
print "run time: ", run_time

print("")
print "Successfully trained and run with balanced accuracy: ", balanced_accuracy

# spio.savemat('baseline_acc.mat', dict(balanced_accuracy=balanced_accuracy))
# spio.savemat('baseline_run_time.mat', dict(run_time=run_time))
#
# print("")
# print "Successfully saved to 'baseline.mat' with shape: ", np.shape(balanced_accuracy)

