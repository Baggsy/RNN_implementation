from pylab import *
from tensorflow.python.keras.models import model_from_config
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import Callback
from tensorflow import keras
import csv
import os


class EarlyStoppingByLossVal(Callback):

    def __init__(self, monitor='loss', value=0.001, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        else:
            if (current < self.value and logs.get('acc') > 0.95) or (epoch > 50 and logs.get('acc') == 0):
                if self.verbose > 0:
                    print "\n----- Stopping at Acc: ", logs.get('acc'), " Loss: ", logs.get('loss'), " -----\n"
                self.model.stop_training = True


class EarlyStoppingByLossVal2(Callback):

    def __init__(self, monitor='acc', value=0.95, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        else:
            if current > self.value or logs.get('loss') < 0.1:
                if self.verbose > 0:
                    print "\n----- Stopping at Acc: ", logs.get('acc'), " Loss: ", logs.get('loss'), " -----\n"
                self.model.stop_training = True


class EarlyStoppingByLossVal3(Callback):

    def __init__(self, monitor='loss', value=0.001, verbose=1):
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
                    print "\n----- Stopping at Acc: ", logs.get('acc'), " Loss: ", logs.get('loss'), " -----\n"
                self.model.stop_training = True



def read_data(n_folds):
    x_train = [[] for _ in xrange(n_folds)]
    x_test = [[] for _ in xrange(n_folds)]
    y_train = [[] for _ in xrange(n_folds)]
    y_test = [[] for _ in xrange(n_folds)]
    x_val = [[] for _ in xrange(n_folds)]
    y_val = [[] for _ in xrange(n_folds)]

    # Naming convention of the files
    start_train = "_train["
    start_test = "_test["
    start_val = "_val["
    file_type_end = "].csv"

    path = "data/"
    for i in range(0, n_folds):
        csv_x_train = path + "x" + start_train + str(i) + file_type_end
        csv_y_train = path + "y" + start_train + str(i) + file_type_end
        csv_x_test = path + "x" + start_test + str(i) + file_type_end
        csv_y_test = path + "y" + start_test + str(i) + file_type_end
        csv_x_val = path + "x" + start_val + str(i) + file_type_end
        csv_y_val = path + "y" + start_val + str(i) + file_type_end

        x_train[i] = np.array(list(csv.reader(open(csv_x_train))))
        y_train[i] = np.array(list(csv.reader(open(csv_y_train)))).astype(int)
        x_test[i] = np.array(list(csv.reader(open(csv_x_test))))
        y_test[i] = np.array(list(csv.reader(open(csv_y_test)))).astype(int)
        x_val[i] = np.array(list(csv.reader(open(csv_x_val))))
        y_val[i] = np.array(list(csv.reader(open(csv_y_val)))).astype(int)

        # Reshape to tensor input shape for LSTM
        x_train[i] = np.reshape(x_train[i], (x_train[i].shape[0], 1, x_train[i].shape[1]))
        x_test[i] = np.reshape(x_test[i], (x_test[i].shape[0], 1, x_test[i].shape[1]))
        x_val[i] = np.reshape(x_val[i], (x_val[i].shape[0], 1, x_val[i].shape[1]))

    return x_train, x_test, x_val, y_val, y_train, y_test


def plot_hist(history, i):
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'+str(i)], loc='upper left')
    # plt.show()
    savefig('loss_function_' + str(i) + 'fold.png')


def clone_model(model, isloaded, layers_to_load, lstm_type, unit, set, custom_objects={}):
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    if isloaded:
        path = "models_saved/Weights_layers:{}_Type:{}_units:{}_Set:{}.h5".format(layers_to_load, lstm_type, unit, set)
        print "cloning weights: {}".format(path)
        clone.load_weights(path, by_name=True)
        # clone.set_weights(model_recovering.get_weights())

    return clone


def find_balanced_accuracy(predicted_labels, y_test):
    a = np.array([predicted_labels[:, 0]]).transpose()
    b = np.array([predicted_labels[:, 1]]).transpose()
    a = np.array(a >= b).astype(int)
    b = np.array(abs(a - 1))
    y_pred = np.concatenate((a, b), axis=1)

    # Calculating the balanced accuracy
    TP = sum(((y_pred == [1, 0])[:, 0]) & (((y_pred == y_test).astype(int))[:, 0])) * 1.0
    TN = sum(((y_pred == [0, 1])[:, 0]) & (((y_pred == y_test).astype(int))[:, 0])) * 1.0
    P = sum(((y_test == [1, 0]).astype(int))[:, 0]) * 1.0
    N = sum(((y_test == [0, 1]).astype(int))[:, 0]) * 1.0
    bal_accuracy = (TP / P + TN / N) / 2.0
    print("TP:", TP)
    print("TN:", TN)
    print("P:", P)
    print("N:", N)
    print "bal_accuracy: ", bal_accuracy

    return bal_accuracy


def train_model(x_train, y_train, x_val, y_val, validation_split, epoch_train, mini_batch_size, shuffle,
                                            x_test, y_test, model, n_folds, learning_rate, optimizer, verbose,
                                            loss_function, metrics, isloaded, n_layers, layers_to_load, type, unit):

    bal_accuracy = [[0 for _ in xrange(n_folds)] for _ in xrange(6)]
    models = [[0 for _ in xrange(n_folds)] for _ in xrange(6)]
    predicted_labels = [0 for _ in xrange(n_folds)]
    models_to_save = [0 for _ in xrange(n_folds)]

    model1 = clone_model(model, isloaded, layers_to_load, type, unit, 0)
    model2 = clone_model(model, isloaded, layers_to_load, type, unit, 1)
    model3 = clone_model(model, isloaded, layers_to_load, type, unit, 2)
    model4 = clone_model(model, isloaded, layers_to_load, type, unit, 3)
    model1.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    model2.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    model3.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    model4.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

    callback = [0 for _ in xrange(6)]

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    EarlyStopping = EarlyStoppingByLossVal(monitor='loss', value=0.001, verbose=verbose)

    callback[0] = [EarlyStoppingByLossVal2(monitor='acc', value=0.95, verbose=verbose)]
    callback[1] = [EarlyStoppingByLossVal3(monitor='loss', value=0.1, verbose=verbose)]
    callback[2] = [EarlyStoppingByLossVal3(monitor='loss', value=0.05, verbose=verbose)]
    callback[3] = [EarlyStoppingByLossVal3(monitor='loss', value=0.01, verbose=verbose)]
    callback[4] = [EarlyStoppingByLossVal3(monitor='loss', value=0.005, verbose=verbose)]
    callback[5] = [EarlyStopping]

    # if isloaded > 1:
    #     shuffle = False

    for callb in callback:
        model1.fit(x_train[0], y_train[0], epochs=epoch_train, batch_size=mini_batch_size, callbacks=callb, verbose=verbose, shuffle=shuffle)
        model2.fit(x_train[1], y_train[1], epochs=epoch_train, batch_size=mini_batch_size, callbacks=callb, verbose=verbose, shuffle=shuffle)
        model3.fit(x_train[2], y_train[2], epochs=epoch_train, batch_size=mini_batch_size, callbacks=callb, verbose=verbose, shuffle=shuffle)
        model4.fit(x_train[3], y_train[3], epochs=epoch_train, batch_size=mini_batch_size, callbacks=callb, verbose=verbose, shuffle=shuffle)
        models[callback.index(callb)][0] = model1
        models[callback.index(callb)][0].set_weights(model1.get_weights())
        models[callback.index(callb)][1] = model2
        models[callback.index(callb)][1].set_weights(model2.get_weights())
        models[callback.index(callb)][2] = model3
        models[callback.index(callb)][2].set_weights(model3.get_weights())
        models[callback.index(callb)][3] = model4
        models[callback.index(callb)][3].set_weights(model4.get_weights())

        # plot_hist(history1, 1)
        # plot_hist(history2, 2)
        # plot_hist(history3, 3)
        # plot_hist(history4, 4)

        # Predicting the test data labels
        predicted_labels[0] = model1.predict(x_val[0])
        predicted_labels[1] = model2.predict(x_val[1])
        predicted_labels[2] = model3.predict(x_val[2])
        predicted_labels[3] = model4.predict(x_val[3])

        for i in range(0, n_folds):
            bal_accuracy[callback.index(callb)][i] = find_balanced_accuracy(predicted_labels[i], y_val[i])
            # print "bal_accuracy[{}]: {}".format(i, bal_accuracy[i])

    print "\nbal_accuracy: ", bal_accuracy, "\n"

    for i in xrange(n_folds):
        temp = [row[i] for row in bal_accuracy]
        index = temp.index(np.max(temp))
        models_to_save[i] = models[index][i]
        print "saving model index: {}".format(index)
        print "saving model set: {}".format(i)
        # print bal_accuracy

    balanced_accuracy = [0 for _ in xrange(n_folds)]
    for i in range(0, n_folds):
        balanced_accuracy[i] = find_balanced_accuracy(models_to_save[i].predict(x_test[i]), y_test[i])

    average_balanced_accuracy = mean(balanced_accuracy)

    for model2 in models_to_save:
        path_model = "models_saved/Models_layers:{}_Type:{}_units:{}_Set:{}.h5".format(n_layers, type, unit,
                                                                        models_to_save.index(model2))
        path_weight = "models_saved/Weights_layers:{}_Type:{}_units:{}_Set:{}.h5".format(n_layers, type, unit,
                                                                        models_to_save.index(model2))

        if os.path.isfile(path_model):
            os.remove(path_model)
            os.remove(path_weight)

        model2.save(path_model)
        model2.save_weights(path_weight)


    print "Average balanced_accuracy: ", average_balanced_accuracy

    return average_balanced_accuracy
