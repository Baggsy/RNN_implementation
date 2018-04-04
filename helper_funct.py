from pylab import *
from tensorflow.python.keras.models import model_from_config
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import Callback
import csv


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=0.0005, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        else:
            if current < self.value:
                if self.verbose > 0:
                    print("\nEpoch %05d: early stopping THR\n" % epoch)
                self.model.stop_training = True


def read_data(n_folds):
    x_train = [[] for _ in xrange(n_folds)]
    x_test = [[] for _ in xrange(n_folds)]
    y_train = [[] for _ in xrange(n_folds)]
    y_test = [[] for _ in xrange(n_folds)]

    # Naming convention of the files
    start_train = "_train["
    start_test = "_test["
    file_type_end = "].csv"

    path = "data/"
    for i in range(0, n_folds):
        csv_x_train = path + "x" + start_train + str(i) + file_type_end
        csv_y_train = path + "y" + start_train + str(i) + file_type_end
        csv_x_test = path + "x" + start_test + str(i) + file_type_end
        csv_y_test = path + "y" + start_test + str(i) + file_type_end

        x_train[i] = np.array(list(csv.reader(open(csv_x_train))))
        y_train[i] = np.array(list(csv.reader(open(csv_y_train)))).astype(int)
        x_test[i] = np.array(list(csv.reader(open(csv_x_test))))
        y_test[i] = np.array(list(csv.reader(open(csv_y_test)))).astype(int)

        # Reshape to tensor input shape for LSTM
        x_train[i] = np.reshape(x_train[i], (x_train[i].shape[0], 1, x_train[i].shape[1]))
        x_test[i] = np.reshape(x_test[i], (x_test[i].shape[0], 1, x_test[i].shape[1]))

    return x_train, x_test, y_train, y_test


def plot_hist(history, i):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    savefig('loss_function_' + str(i) + 'fold.png')


def clone_model(model, custom_objects={}):
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    # clone.set_weights(model.get_weights())
    return clone


def train_model(x_train, y_train, validation_split, epoch_train, mini_batch_size,
                callbacks, x_test, y_test, model, n_folds, learning_rate):
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
        a = np.array([predicted_labels[i][:, 0]]).transpose()
        b = np.array([predicted_labels[i][:, 1]]).transpose()
        a = np.array(a >= b).astype(int)
        b = np.array(abs(a-1))
        y_pred = np.concatenate((a, b), axis=1)

        # Calculating the balanced accuracy
        TP = sum(((y_pred == [1, 0])[:, 0]) & (((y_pred == y_test[i]).astype(int))[:, 0]))*1.0
        TN = sum(((y_pred == [0, 1])[:, 0]) & (((y_pred == y_test[i]).astype(int))[:, 0]))*1.0
        P = sum(((y_test[i] == [1, 0]).astype(int))[:, 0])*1.0
        N = sum(((y_test[i] == [0, 1]).astype(int))[:, 0])*1.0
        bal_accuracy[i] = (TP/P + TN/N)/2.0
        print("TP:", TP)
        print("TN:", TN)
        print("P:", P)
        print("N:", N)
        print "bal_accuracy[i]: ", bal_accuracy[i]

    return bal_accuracy
