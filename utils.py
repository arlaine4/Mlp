import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def get_arguments():
    args = argparse.ArgumentParser()
    args.add_argument("data_path", help='file path to data')
    args.add_argument("mode", help='mode to run the program with')
    options = args.parse_args()
    return options

def check_arguments(args):
    if args.mode != 'prediction' and args.mode != 'learning':
        return True, "You need to specify a mode to run the program, either 'prediction' or 'training'"
    return False, ""

def normalize_data(data):
    """
    Data normalization, scaling data
    to make life easier for the model,
    its a common thing to do in ML / DL
    """
    mean = np.mean(data, axis = 0)
    scale = np.std(data - mean, axis = 0)
    return (data - mean) / scale

def define_layers_sizes(X, Y):
    """
    Simple method to define the layers sizes
    we init input to the input_shape, hidden
    to an arbitrary 20 neurons, and outputs
    to outputs_shape
    """
    s_x = X.shape[0]
    s_hidden = 20
    s_y = Y.shape[0]
    return s_x, s_hidden, s_y

def initialize_model_structure(s_x, s_h, s_y):
    """
    Initializing model structure, layers etc
        W1 = first layer of weights
        b1 = first layer of biases
        W2 = second layer of weights
        b2 = second layer of biases

        we use a numpy random seed to get the same
        results when generating data with randn

        we initialize biases with 0 not
        like the weights
    """
    np.random.seed(42)
    struct = {"W1" : np.random.randn(s_h, s_x) * 0.01,
                "b1" : np.zeros((s_h, 1)),
                "W2" : np.random.randn(s_y, s_h) * 0.01,
                "b2" : np.zeros((s_y, 1))}
    return struct

def get_data(file_path):
    """
    Getting data from data_path
        X_train = training inputs
        Y_train = training labels associated with X_train
        X_test = test inputs
        Y_test = test labels associated with X_test
    """
    try:
        data = pd.read_csv(file_path, header=None)
    except:
        sys.exit("This csv is not valid")
    Y = np.array([data[1]])
    Y = np.where(Y == "M", 1, 0)
    X = np.array(data.iloc[:, 2:])
    X = normalize_data(X)
    # train_test_split is a method that splits a dataset into subparts
    # => train_inputs, train_labels, test_inputs and test_labels
    # ==> inputs is just what you feed the model and labels are
    # ==> the results associated with each input
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y.T, test_size = 0.25, random_state = 1)
    X_train = X_train.T
    X_test = X_test.T
    Y_train = Y_train.reshape((1, len(Y_train)))
    Y_test = Y_test.reshape((1, len(Y_test)))
    return X_train, X_test, Y_train, Y_test
