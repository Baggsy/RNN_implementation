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

# print(np.shape(x))
# print(np.shape(y))
# print(np.shape(y_set_labels))
# print('length x: ', len(x))
# print('length y: ', len(y))
# print('length y_s: ', len(y_set_labels))

# x_train = [x_data[2,:,:]; x_data[3,:,:]; x_data[5,:,:]; x_data[6,:,:]; x_data[7,:,:]]
# x_val = [x_data[8,:,:]]
# x_test = [x_data[1,:,:]; x_data[4,:,:]]
#
# y_train = [y_data[2,:,:]; y_data[3,:,:]; y_data[5,:,:]; y_data[6,:,:]; y_data[7,:,:]]
# y_val = [y_data[8,:,:]]
# y_test = [y_data[1,:,:]; y_data[4,:,:]]


BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"

n_GPUS = 1

def data_type():
    return tf.float32


def MSI_data_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).

    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.

    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y


class MSI_Input(object):
    """The input data."""
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = MSI_data_producer(data, batch_size, num_steps, name=name)


class MSI_Model(object):
    """The MSI model."""

    def __init__(self, is_training, config, input_):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_rnn_graph(inputs, config, is_training)

        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self._cost = tf.reduce_sum(loss)
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _build_rnn_graph(self, inputs, config, is_training):
        if config.rnn_mode == CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        """Build the inference graph using CUDNN cell."""
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=config.num_layers,
            num_units=config.hidden_size,
            input_size=config.hidden_size,
            dropout=1 - config.keep_prob if is_training else 0)
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
            "lstm_params",
            initializer=tf.random_uniform(
                [params_size_t], -config.init_scale, config.init_scale),
            validate_shape=False)
        c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                     tf.float32)
        h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                     tf.float32)
        self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, config.hidden_size])
        return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

    def _get_lstm_cell(self, config, is_training):
        if config.rnn_mode == BASIC:
            return tf.contrib.rnn.BasicLSTMCell(
                config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                reuse=not is_training)
        if config.rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(
                config.hidden_size, forget_bias=0.0)
        raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state
        # Simplified version of tf.nn.static_rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        # outputs, state = tf.nn.static_rnn(cell, inputs,
        #                                   initial_state=self._initial_state)
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = util.with_prefix(self._name, "initial")
        self._final_state_name = util.with_prefix(self._name, "final")
        util.export_state_tuples(self._initial_state, self._initial_state_name)
        util.export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self):
        """Imports ops from collections."""
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
                    self._cell,
                    self._cell.params_to_canonical,
                    self._cell.canonical_to_params,
                    rnn_params,
                    base_variable_scope="Model/RNN")
                tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
        num_replicas = n_GPUS if self._name == "Train" else 1
        self._initial_state = util.import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = util.import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK

class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


def get_config(model_type):
    """Get model config."""
    config = None
    if model_type == "small":
        config = SmallConfig()
    elif model_type == "medium":
        config = MediumConfig()
    elif model_type == "large":
        config = LargeConfig()
    elif model_type == "test":
        config = TestConfig()
    return config


n_test = int(round(len(x) * (1 - 0.2)))
n_val = int(round(len(x[:n_test, :]) * (1 - 0.2)))

x_train = x[1:n_val, :]
x_val = x[n_val: n_test, :]
x_test = x[n_test:, :]

y_train = y[1:n_val]
y_val = y[n_val: n_test]
y_test = y[n_test:]


model_type = "small"
config = get_config(model_type)
eval_config = get_config(model_type)
eval_config.batch_size = 1
eval_config.num_steps = 1

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
        train_input = MSI_Input(config=config, data=x_train, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = MSI_Model(is_training=True, config=config, input_=train_input)
        tf.summary.scalar("Training Loss", m.cost)
        tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
        valid_input = MSI_Input(config=config, data=x_val, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = MSI_Model(is_training=False, config=config, input_=valid_input)
        tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
        test_input = MSI_Input(config=eval_config, data=x_test, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = MSI_Model(is_training=False, config=eval_config, input_=test_input)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
        model.export_ops(name)
    metagraph = tf.train.export_meta_graph()

    soft_placement = False
    if n_GPUS > 1:
        soft_placement = True
        util.auto_parallel(metagraph, m)


if __name__ == "__main__":
    tf.app.run()
