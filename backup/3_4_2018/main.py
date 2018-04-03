#!python

import scipy.io as spio
from pylab import *
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, LSTM, Bidirectional
from tensorflow.python.keras.optimizers import Adam
from read_data import read_data

# HYPER Parameters
mini_batch_size = 32
embedding_size = 8
learning_rate = 0.001
epoch_train = 300
validation_split = 0.05

# Model parameters
# units = [5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 1000, 2000, 10000, 20000, 50000]
# layers = range(1, 31)
units = [500, 1000]
layers = [1, 5, 10]
lstm_type = ['LSTM', 'Bidirectional']

# Data specific parameters
n_sets = 8
n_folds = 4

# initialing the 3D output array
balanced_accuracy = np.zeros((len(units), len(layers), len(lstm_type)))
bal_accuracy = np.zeros(n_folds)

# Reading the data from the read_data function
x_train, x_test, y_train, y_test = read_data(n_sets, n_folds)

# in_shape defines the input shape of the LSTM modules
in_shape = len(x_train[0][0][0])

for k in range(0, len(units)):
    for m in range(0, len(layers)):
        for l in range(0, len(lstm_type)):
            model = Sequential()
            unit = units[k]
            n_layer = layers[m]

            # Adding the LSTM layers or the Bidirectional LSTM modules
            if l == 0:
                for j in range(0, n_layer-1):
                    model.add(LSTM(units=unit, input_shape=(1, in_shape),  return_sequences=True, bias_initializer='random_normal'))

                model.add(LSTM(units=unit, input_shape=(1, in_shape),  bias_initializer='random_normal'))
            else:
                for j in range(0, n_layer-1):
                    model.add(Bidirectional(LSTM(units=unit, return_sequences=True, bias_initializer='random_normal'), input_shape=(1, in_shape), merge_mode='concat'))

                model.add(Bidirectional(LSTM(units=unit, bias_initializer='random_normal'), input_shape=(1, in_shape), merge_mode='concat'))

            # Adding the rest of the network's components
            model.add(Dense(units=2, bias_initializer='random_normal'))
            model.add(Activation(activation='softmax'))
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

            model.summary()

            for i in range(0, n_folds):
                # Training the model
                print("unit: ", unit)
                print("n_layer: ", n_layer)
                print("lstm_type: ", lstm_type[l])
                print("fold: ", i)
                model.fit(x_train[i], y_train[i], validation_split=validation_split, epochs=epoch_train, batch_size=mini_batch_size)

                # Predicting the test data labels
                predicted_labels = model.predict(x_test[i])
                y_pred = (predicted_labels >= 0.5).astype(int)

                # Calculating the balanced accuracy
                TP = sum(((y_pred == [1, 0])[:, 0]) & ((y_pred == y_test[i])[:, 0]))
                TN = sum(((y_pred == [0, 1])[:, 0]) & ((y_pred == y_test[i])[:, 0]))
                P = sum((y_test[i] == [1, 0])[:, 0])
                N = sum((y_test[i] == [0, 1])[:, 0])
                bal_accuracy[i] = (TP/P + TN/N)/2.0

            # Clearing the variables for safe operations
            del model

            # Saving the balanced accuracy over the 4 folds
            balanced_accuracy[k, m, l] = mean(bal_accuracy[:])


print("")
print("Successfully trained and run")

spio.savemat('balanced_accuracy_reduced.mat', dict(balanced_accuracy=balanced_accuracy))

print("")
print "Successfully saved to 'balanced_accuracy.mat' with shape: ", np.shape(balanced_accuracy)

