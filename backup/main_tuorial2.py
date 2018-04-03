#!python
#!/usr/bin/env python

import tensorflow as tf
import scipy.io as spio
import h5py
import numpy as np
import tables
import pandas as pd
import csv
from pylab import *
from matplotlib import *
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lstm_predictor import generate_data, load_csvdata, lstm_model

import util

x = np.array(list(csv.reader(open("csv_data.csv"))))
y = np.array(list(csv.reader(open("csv_truth.csv"))))
y_set_labels = np.array(list(csv.reader(open("csv_truth_labels.csv"))))

# x_train = [x_data[2,:,:]; x_data[3,:,:]; x_data[5,:,:]; x_data[6,:,:]; x_data[7,:,:]]
# x_val = [x_data[8,:,:]]
# x_test = [x_data[1,:,:]; x_data[4,:,:]]
#
# y_train = [y_data[2,:,:]; y_data[3,:,:]; y_data[5,:,:]; y_data[6,:,:]; y_data[7,:,:]]
# y_val = [y_data[8,:,:]]
# y_test = [y_data[1,:,:]; y_data[4,:,:]]

n_test = int(round(len(x) * (1 - 0.2)))
n_val = int(round(len(x[:n_test, :]) * (1 - 0.2)))

x_train = x[1:n_val, :]
x_val = x[n_val: n_test, :]
x_test = x[n_test:, :]

y_train = y[1:n_val]
y_val = y[n_val: n_test]
y_test = y[n_test:]

batch_size = 32
num_layers = 1

class lstm_size(object):
    """lstm config."""
    hidden_size = 200
    forget_bias=0.0
    state_is_tuple=True
    reuse = False


def make_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    return cell


lstm = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(num_layers)], state_is_tuple=True)

# Initial state of the LSTM memory.
hidden_state = tf.zeros([batch_size, lstm.state_size])
current_state = tf.zeros([batch_size, lstm.state_size])
state = hidden_state, current_state
probabilities = []
loss = 0.0

for current_batch_of_words in words_in_dataset:
    # The value of state is updated after processing each batch of words.
    output, state = lstm(current_batch_of_words, state)

    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities.append(tf.nn.softmax(logits))
    loss += loss_function(probabilities, target_words)




if __name__ == "__main__":
    tf.app.run()
