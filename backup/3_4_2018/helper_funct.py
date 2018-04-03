from pylab import *
from tensorflow.python.keras.models import Sequential, model_from_config
from tensorflow.python.keras.layers import Dense, Activation, LSTM
from tensorflow.python.keras.optimizers import RMSprop, Adam
from tensorflow import keras

def plot_hist(history, i):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    savefig('loss_function_' + str(i) + 'fold.png')


def batch_generator(batch_size, sequence_length, x_train, y_train, x_len, in_length, out_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, in_length)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, out_length)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(x_len - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train[idx:idx+sequence_length]
            y_batch[i] = y_train[idx:idx+sequence_length]

        yield (x_batch, y_batch)

def prepare_sequences(x_train, y_train, window_length):
    windows = []
    windows_y = []
    for i, sequence in enumerate(x_train):
        len_seq = len(sequence)
        for window_start in range(0, len_seq - window_length + 1):
            # print "here: ", i, " ", len(sequence)
            window_end = window_start + window_length
            window = sequence[window_start:window_end]
            windows.append(window)
            windows_y.append(y_train[i])
    return np.array(windows), np.array(windows_y)


def clone_model(model, custom_objects={}):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone

def train_model(x_train, y_train, validation_split, epoch_train, mini_batch_size,
                callbacks, x_test, y_test, model, n_folds, learning_rate):
    # model = Sequential()
    # # Adding the LSTM layers or the Bidirectional LSTM modules
    # model.add(LSTM(units=units, input_shape=(1, in_shape), bias_initializer='random_normal'))
    #
    # # Adding the rest of the network's components
    # model.add(Dense(units=2, bias_initializer='random_normal'))
    # model.add(Activation(activation='softmax'))
    # model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['accuracy'])
    #
    # model.summary()

    bal_accuracy = np.zeros(n_folds)
    predicted_labels = [0 for _ in xrange(n_folds)]

    model1 = clone_model(model)
    model2 = clone_model(model)
    model3 = clone_model(model)
    model4 = clone_model(model)
    model1.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['accuracy'])
    model2.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['accuracy'])
    model3.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['accuracy'])
    model4.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['accuracy'])

    # model1.set_weights(model.load_weights)
    # model2.set_weights(model.load_weights)
    # model3.set_weights(model.load_weights)
    # model4.set_weights(model.load_weights)

    history1 = model1.fit(x_train[0], y_train[0], validation_split=validation_split, epochs=epoch_train, batch_size=mini_batch_size, callbacks=callbacks)
    history2 = model2.fit(x_train[1], y_train[1], validation_split=validation_split, epochs=epoch_train, batch_size=mini_batch_size, callbacks=callbacks)
    history3 = model3.fit(x_train[2], y_train[2], validation_split=validation_split, epochs=epoch_train, batch_size=mini_batch_size, callbacks=callbacks)
    history4 = model4.fit(x_train[3], y_train[3], validation_split=validation_split, epochs=epoch_train, batch_size=mini_batch_size, callbacks=callbacks)

    plot_hist(history1, 1)
    plot_hist(history2, 2)
    plot_hist(history3, 3)
    plot_hist(history4, 4)

    # Predicting the test data labels
    predicted_labels[0] = model1.predict(x_test[0])
    predicted_labels[1] = model2.predict(x_test[1])
    predicted_labels[2] = model3.predict(x_test[2])
    predicted_labels[3] = model4.predict(x_test[3])

    for i in range(0, n_folds):
        print i
        print np.shape(x_train[i])
        print np.shape(y_train[i])
        print np.shape(x_test[i])
        print np.shape(y_test[i])
        print "x_train[i][:30]: ", x_train[i][:30]
        print "y_train[i][:30]: ", y_train[i][:30]
        print "x_test[i][:30]: ", x_test[i][:30]
        print "y_test[i][:30]: ", y_test[i][:30]

        print "np.shape(predicted_labels): ", np.shape(predicted_labels[i])
        print "np.shape(y_test[i]: ", np.shape(y_test[i])
        a = np.array([predicted_labels[i][:, 0]]).transpose()
        b = np.array([predicted_labels[i][:, 1]]).transpose()
        a = np.array(a >= b).astype(int)
        b = np.array(abs(a-1))
        y_pred = np.concatenate((a, b), axis=1)

        # Calculating the balanced accuracy
        TP = sum(((y_pred == [1, 0])[:, 0]) & (((y_pred == y_test[i]).astype(int))[:, 0]))*1.0
        TN = sum(((y_pred == [0, 1])[:, 0]) & (((y_pred == y_test[i]).astype(int))[:, 0]))*1.0
        P = sum((y_test[i] == [1, 0])[:, 0])*1.0
        N = sum((y_test[i] == [0, 1])[:, 0])*1.0
        bal_accuracy[i] = (TP/P + TN/N)/2.0
        print("TP:", TP)
        print("TN:", TN)
        print("P:", P)
        print("N:", N)
        print "bal_accuracy[i]: ", bal_accuracy[i]


        # del history
        #
        # initializing the weights
        # model.reset_states()
        # weights = model.get_weights()
        # weights = [ones(w.shape)*0.5 for w in weights]
        # model.set_weights(weights)
        # del model

    return bal_accuracy
