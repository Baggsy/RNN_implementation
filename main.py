#!python

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, LSTM, Bidirectional
from tensorflow import keras
from helper_funct import *
import time


# HYPER Parameters
mini_batch_size = 32
embedding_size = 8
learning_rate = 0.001
epoch_train = 300  # maximum repetitions
validation_split = 0.05
optimizer = RMSprop(lr=learning_rate)
metrics = ['accuracy']
bias_init = 'random_normal'

# Model parameters
units = [500, 1000]
layers = [1, 5, 10]
lstm_type = ['LSTM', 'Bidirectional']
activation = 'softmax'
loss_function = 'binary_crossentropy'
merge_mode = 'concat'

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
balanced_accuracy = np.zeros((len(units), len(layers), len(lstm_type)))

# Reading the data from the read_data function
x_train, x_test, y_train, y_test = read_data(n_folds)

# in_shape defines the input shape of the LSTM modules
in_shape = len(x_train[0][0][0])  # data length variable for the input tensor
start_time = time.time()

for unit in units:
    for n_layers in layers:
        for type in lstm_type:

            start_time2 = time.time()
            model = Sequential()
            # Adding the LSTM layers or the Bidirectional LSTM modules
            if type == 'LSTM':
                for j in range(0, n_layers-1):
                    model.add(LSTM(units=unit, input_shape=(time_steps, in_shape), return_sequences=True, bias_initializer=bias_init))
                model.add(LSTM(units=unit, input_shape=(time_steps, in_shape), bias_initializer=bias_init))
            else:
                for j in range(0, n_layers-1):
                    model.add(Bidirectional(LSTM(units=unit, return_sequences=True, bias_initializer=bias_init), input_shape=(time_steps, in_shape), merge_mode=merge_mode))
                model.add(Bidirectional(LSTM(units=unit, bias_initializer=bias_init), input_shape=(time_steps, in_shape), merge_mode=merge_mode))

            # Adding the rest of the network's components
            model.add(Dense(units=num_classes, bias_initializer=bias_init))
            model.add(Activation(activation=activation))
            model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
            model.summary()

            # training of the model
            bal_accuracy = train_model(x_train, y_train, validation_split, epoch_train, mini_batch_size,
                                       callbacks, x_test, y_test, model, n_folds, learning_rate)

            model.reset_states()
            model.set_weights([ones(w.shape)*0.5 for w in model.get_weights()])
            del model

            # Saving the balanced accuracy over the 4 folds
            balanced_accuracy[units.index(unit), layers.index(n_layers), lstm_type.index(type)] = mean(bal_accuracy[:])

            run_time2 = time.time() - start_time2
            print "run time of units ", unit, " n_layers ", n_layers, " of type ", type, ": ", run_time2

run_time = time.time() - start_time
print "balanced_accuracy: ", balanced_accuracy
print "Total run time: ", run_time

balanced_accuracy_final = balanced_accuracy.mean()

print("")
print "Successfully trained and run with balanced_accuracy_final: ", balanced_accuracy_final

# spio.savemat('baseline_acc.mat', dict(balanced_accuracy=balanced_accuracy))
# spio.savemat('baseline_run_time.mat', dict(run_time=run_time))
#
# print("")
# print "Successfully saved to 'baseline.mat' with shape: ", np.shape(balanced_accuracy)

