#!python

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation, LSTM, Bidirectional
from tensorflow.python.keras import metrics
from helper_funct import *
import time
import os

# TODO: gradient back propagation
#       freeze weights and load them for more layers. Train stack
#       use 100 units to find best method.
#       Camelyon17 images. Use CNN network
#       combine image and MSI data. Image captioning
#       stateful = true, or return sequences in last layer

# HYPER Parameters
mini_batch_size = 32
embedding_size = 8
learning_rate = 0.005
epoch_train = 300  # maximum repetitions
validation_split = 0.05
optimizer_str = "RMSprop"
optimizer = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0) # keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True) #
metrics = ['accuracy', 'mae']
bias_init = 'he_normal'  # It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) where  fan_in is the number of input units in the weight tensor.
kernel_init = 'he_normal'
weight_init = 'he_normal'
use_bias = True

# Model parameters
units = [500, 1000]
layers = [5, 10]
lstm_type = ['LSTM', 'Bidirectional']
activation = 'softmax'
loss_function = 'binary_crossentropy'
merge_mode = 'average' # 'concat'


units = [100]
layers = [1]
lstm_type = ['LSTM']

file = open("results.txt", "a")
file.write("mini_batch_size: {} learning_rate: {} optimizer: {}".format(mini_batch_size, learning_rate, optimizer_str))


# Data specific parameters
n_sets = 8
n_folds = 4
num_classes = 2
time_steps = 1

# initialing the output array
bal_accuracy = np.zeros(n_folds)
balanced_accuracy = np.zeros((len(units), len(layers), len(lstm_type), n_folds))
run_time2 = np.zeros((len(units), len(layers), len(lstm_type)))

# Reading the data from the read_data function
x_train, x_test, y_train, y_test = read_data(n_folds)

# in_shape defines the input shape of the LSTM modules
in_shape = len(x_train[0][0][0])  # data length variable for the input tensor
start_time = time.time()

# ______________________________________________________


# #  ->>>>>>>>>>>>>>>>>>>>> stateful make all equally batched divisible + specify shuffle=False when calling fit().
# for unit in units:
#     for n_layers in layers:
#         for type in lstm_type:
#             model = Sequential()
#             model.add(LSTM(units=unit, input_shape=(time_steps, in_shape), bias_initializer=bias_init,
#                             kernel_initializer=kernel_init, recurrent_initializer=weight_init, stateful=True, batch_size=mini_batch_size,
#                             use_bias=use_bias))
#
#             model.add(Dense(units=num_classes))
#             model.add(Activation(activation=activation))
#             model.summary()
#
#             # training of the model
#             bal_accuracy = train_model(x_train, y_train, validation_split, epoch_train, mini_batch_size,
#                                        x_test, y_test, model, n_folds, learning_rate, optimizer,
#                                        loss_function, metrics)
#

# ______________________________________________________

for unit in units:
    for n_layers in layers:
        for type in lstm_type:
            print "unit: ", unit
            print "n_layers: ", n_layers
            print "type: ", type
            start_time2 = time.time()
            model = Sequential()
            # Adding the LSTM layers or the Bidirectional LSTM modules
            if type == 'LSTM':
                if n_layers > 1:
                    model.add(LSTM(units=unit, input_shape=(time_steps, in_shape), return_sequences=True,
                                   bias_initializer=bias_init, kernel_initializer=kernel_init,
                                   recurrent_initializer=weight_init, use_bias=use_bias, name="LSTM_1"))
                    for j in range(2, n_layers):
                        model.add(LSTM(units=unit, return_sequences=True, bias_initializer=bias_init,
                                       kernel_initializer=kernel_init, recurrent_initializer=weight_init,
                                       use_bias=use_bias, name = "LSTM_{}".format(j)))
                    model.add(LSTM(units=unit, bias_initializer=bias_init, kernel_initializer=kernel_init,
                                   recurrent_initializer=weight_init, use_bias=use_bias, name="LSTM_{}".format(n_layers)))
                else:
                    model.add(LSTM(units=unit, input_shape=(time_steps, in_shape), bias_initializer=bias_init,
                                   kernel_initializer=kernel_init, recurrent_initializer=weight_init,
                                   use_bias=use_bias, name="LSTM_1"))
            # else:
            #     if n_layers > 1:
            #         model.add(Bidirectional(LSTM(units=unit, return_sequences=True, bias_initializer=bias_init, kernel_initializer=kernel_init, recurrent_initializer=weight_init, use_Bias=use_bias), input_shape=(time_steps, in_shape), merge_mode=merge_mode))
            #         for j in range(1, n_layers-1):
            #             model.add(Bidirectional(LSTM(units=unit, return_sequences=True, bias_initializer=bias_init, kernel_initializer=kernel_init, recurrent_initializer=weight_init, use_Bias=use_bias), merge_mode=merge_mode))
            #         model.add(Bidirectional(LSTM(units=unit, bias_initializer=bias_init, kernel_initializer=kernel_init, recurrent_initializer=weight_init, use_Bias=use_bias), merge_mode=merge_mode))
            #     else:
            #         model.add(Bidirectional(LSTM(units=unit, bias_initializer=bias_init, kernel_initializer=kernel_init, recurrent_initializer=weight_init, use_Bias=use_bias), input_shape=(time_steps, in_shape), merge_mode=merge_mode))

            i = n_layers - 1
            isloaded = False
            while i > 0 and not isloaded:
                file_name = "Weights_layers:{}_Type:{}_units:{}_finalLayerSize:{}.h5".format(i, lstm_type, units, n_layers)
                if os.path.isfile(file_name):
                    model.load_weights(file_name, by_name=True)
                    isloaded = True
                i -= 1

            if isloaded:
                for layer in model.layers[:i+1]:
                    layer.trainable = False

            # Adding the rest of the network's components
            model.add(Dense(units=num_classes))
            model.add(Activation(activation=activation))
            # model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
            model.summary()

            # training of the model
            bal_accuracy = train_model(x_train, y_train, validation_split, epoch_train, mini_batch_size,
                                       x_test, y_test, model, n_folds, learning_rate, optimizer,
                                       loss_function, metrics)

            # Saving the balanced accuracy over the 4 folds
            balanced_accuracy[units.index(unit), layers.index(n_layers), lstm_type.index(type)] = bal_accuracy
            run_time2[units.index(unit), layers.index(n_layers), lstm_type.index(type)] = time.time() - start_time2
            print "run time of units {} n_layers {} of type {} : {}".format(unit, n_layers, type, run_time2[units.index(unit), layers.index(n_layers), lstm_type.index(type)])

            file.write(" units: {} n_layers: {} type: {} balanced accuracy: {}".format(unit, n_layers, type, bal_accuracy))

            # model.save("Model_layers:{}_Type:{}_units:{}_finalLayerSize:{}.h5".format(n_layers,lstm_type,units, n_layers))
            model.save_weights("Weights_layers:{}_Type:{}_units:{}_finalLayerSize:{}.h5".format(n_layers,lstm_type,units, n_layers))
            del model


# ______________________________________________________
#
# for unit in units:
#     for n_layers in layers:
#         for type in lstm_type:
#             print "unit: ", unit
#             print "n_layers: ", n_layers
#             print "type: ", type
#             start_time2 = time.time()
#             model = Sequential()
#             i = n_layers-1
#             while i > 0:
#                 file_name = "Model_layers:{}_Type:{}_units:{}_finalLayerSize:{}.h5".format(i,lstm_type,units, n_layers)
#                 if os.path.isfile(file_name):
#                     model2 = load_model(file_name)
#                     for layer in model2.layers:
#                         layer.trainable = False
#                     model.add(model2)
#                     break
#                 i -= 1
#
#             for j in range(i, n_layers+1):
#                 # Adding the LSTM layers or the Bidirectional LSTM modules
#                 if type == 'LSTM':
#                     if j == 1:
#                         model.add(LSTM(units=unit, input_shape=(time_steps, in_shape), bias_initializer=bias_init,
#                                        kernel_initializer=kernel_init, recurrent_initializer=weight_init,
#                                        use_bias=use_bias))
#                     else:
#                         model.add(LSTM(units=unit, bias_initializer=bias_init, kernel_initializer=kernel_init,
#                                        recurrent_initializer=weight_init, use_bias=use_bias))
#                 else:
#                     if j == 1:
#                         model.add(Bidirectional(
#                             LSTM(units=unit, bias_initializer=bias_init, kernel_initializer=kernel_init,
#                                  recurrent_initializer=weight_init, use_Bias=use_bias),
#                             input_shape=(time_steps, in_shape), merge_mode=merge_mode))
#                     else:
#                         model.add(Bidirectional(
#                             LSTM(units=unit, bias_initializer=bias_init, kernel_initializer=kernel_init,
#                                  recurrent_initializer=weight_init, use_Bias=use_bias), merge_mode=merge_mode))
#
#             # Adding the rest of the network's components
#             model.add(Dense(units=num_classes))
#             model.add(Activation(activation=activation))
#             model.summary()
#
#             # training of the model
#             bal_accuracy = train_model(x_train, y_train, validation_split, epoch_train, mini_batch_size,
#                                        x_test, y_test, model, n_folds, learning_rate, optimizer,
#                                        loss_function, metrics)
#
#             model.reset_states()
#             model.set_weights([ones(w.shape)*0.5 for w in model.get_weights()])
#             del model
#
#             # Saving the balanced accuracy over the 4 folds
#             balanced_accuracy[units.index(unit), layers.index(n_layers), lstm_type.index(type)] = bal_accuracy
#             # print "average accuracy: ", balanced_accuracy[units.index(unit), layers.index(n_layers), lstm_type.index(type)]
#             run_time2[units.index(unit), layers.index(n_layers), lstm_type.index(type)] = time.time() - start_time2
#             print "run time of units ", unit, " n_layers ", n_layers, " of type ", type, ": ", run_time2[units.index(unit), layers.index(n_layers), lstm_type.index(type)]
#
#             file.write(" units: {} n_layers: {} type: {} balanced accuracy: {}".format(unit, n_layers, type, bal_accuracy))
#
#
#
#             model.save("Model_layers:{}_Type:{}_units:{}_finalLayerSize:{}.h5".format(i,lstm_type,units, n_layers))
#             model.save_weights("Weights_layers:{}_Type:{}_units:{}_finalLayerSize:{}.h5".format(i,lstm_type,units, n_layers))
#
# run_time = time.time() - start_time
#
# print ""
# print "balanced_accuracy: "
# print balanced_accuracy
# print "run_time2: "
# print run_time2
# # print ""
#
# print "Total run time: ", run_time
#
# balanced_accuracy_final = balanced_accuracy.mean()
#
# print("")
# print "Successfully trained and run with balanced_accuracy_final: ", balanced_accuracy_final

file.write("time: {}\n\n".format(run_time))
file.close()


