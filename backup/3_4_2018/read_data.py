from pylab import *
import csv

def read_data(n_sets, n_folds):
    x = [[] for _ in xrange(n_sets)]
    y = [[] for _ in xrange(n_sets)]
    x_train = [[] for _ in xrange(n_folds)]
    x_test = [[] for _ in xrange(n_folds)]
    y_train = [[] for _ in xrange(n_folds)]
    y_test = [[] for _ in xrange(n_folds)]

    # Naming convention of the files
    file_str_data = "csv_training_cells_"
    file_type = ".csv"
    file_str_truth = "csv_truth_cells_"

    # Reading the data files
    for i in range(0, n_sets):
        file_name_data = file_str_data + str(i+1) + file_type
        file_name_truth = file_str_truth + str(i+1) + file_type
        x[i] = np.array(list(csv.reader(open(file_name_data))))
        y[i] = np.array(list(csv.reader(open(file_name_truth))))

    # Onehotencoding formatting
    y_OneHotEncoded = [[] for _ in xrange(n_sets)]
    for i in range(0, n_sets):
        y_OneHotEncoded[i] = zeros([len(y[i]), 2])
        a = np.array(y[i] == '1').astype(int)
        b = np.array(y[i] == '2').astype(int)
        y_OneHotEncoded[i] = np.concatenate((a, b), axis=1)

    # 1: Training: L2, L3, L5, L6, L7, L8
    #    Test: L1, L4
    x_train[0] = np.concatenate((x[1], x[2], x[4], x[5], x[6], x[7]), axis=0)
    x_test[0] = np.concatenate((x[0], x[3]), axis=0)
    y_train[0] = np.concatenate((y_OneHotEncoded[1], y_OneHotEncoded[2], y_OneHotEncoded[4], y_OneHotEncoded[5], y_OneHotEncoded[6], y_OneHotEncoded[7]), axis=0)
    y_test[0] = np.concatenate((y_OneHotEncoded[0], y_OneHotEncoded[3]), axis=0)

    # 2: Training: L1, L3, L4, L5, L6, L8
    #    Test: L2, L7
    x_train[1] = np.concatenate((x[0], x[2], x[3], x[4], x[5], x[7]), axis=0)
    x_test[1] = np.concatenate((x[1], x[6]), axis=0)
    y_train[1] = np.concatenate((y_OneHotEncoded[0], y_OneHotEncoded[2], y_OneHotEncoded[3], y_OneHotEncoded[4], y_OneHotEncoded[5], y_OneHotEncoded[7]), axis=0)
    y_test[1] = np.concatenate((y_OneHotEncoded[1], y_OneHotEncoded[6]), axis=0)

    # 3: Training: L1, L2, L3, L4, L5, L7
    #    Test: L6, L8
    x_train[2] = np.concatenate((x[0], x[1], x[2], x[3], x[4], x[6]), axis=0)
    x_test[2] = np.concatenate((x[5], x[7]), axis=0)
    y_train[2] = np.concatenate((y_OneHotEncoded[0], y_OneHotEncoded[1], y_OneHotEncoded[2], y_OneHotEncoded[3], y_OneHotEncoded[4], y_OneHotEncoded[6]), axis=0)
    y_test[2] = np.concatenate((y_OneHotEncoded[5], y_OneHotEncoded[7]), axis=0)

    # 4: Training: L1, L2, L4, L6, L7, L8
    #    Test: L3, L5
    x_train[3] = np.concatenate((x[0], x[1], x[3], x[5], x[6], x[7]), axis=0)
    x_test[3] = np.concatenate((x[2], x[4]), axis=0)
    y_train[3] = np.concatenate((y_OneHotEncoded[0], y_OneHotEncoded[1], y_OneHotEncoded[3], y_OneHotEncoded[5], y_OneHotEncoded[6], y_OneHotEncoded[7]), axis=0)
    y_test[3] = np.concatenate((y_OneHotEncoded[2], y_OneHotEncoded[4]), axis=0)

    for i in range(0, n_folds):
        x_train[i] = np.reshape(x_train[i], (x_train[i].shape[0], 1, x_train[i].shape[1]))
        x_test[i] = np.reshape(x_test[i], (x_test[i].shape[0], 1, x_test[i].shape[1]))

    return x_train, x_test, y_train, y_test
