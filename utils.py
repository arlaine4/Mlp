import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import mlp_maths as mlp

def get_arguments():
    args = argparse.ArgumentParser()
    args.add_argument("data_path", help='file path to data')
    args.add_argument("mode", help='mode to run the program with')
    args.add_argument("-erl", "--erl", action='store_true', help='use early stopping')
    args.add_argument("-plt", "--plt", action='store_true', help='plot learning curves')
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
    np.random.seed(100)
    struct = {"W1" : np.random.randn(s_h, s_x) * 0.01,
                "b1" : np.zeros((s_h, 1)),
                "W2" : np.random.randn(s_y, s_h) * 0.01,
                "b2" : np.zeros((s_y, 1))}
    return struct

def get_data(file_path, prediction_mode = False):
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
    if not prediction_mode:
        Y = np.array([data[1]])
        Y = np.where(Y == "M", 1, 0)
        X = np.array(data.iloc[:, 2:])
        X = normalize_data(X)
        # train_test_split is a method that splits a dataset into subparts
        # => train_inputs, train_labels, test_inputs and test_labels
        # ==> inputs is just what you feed the model and labels are
        # ==> the results associated with each input
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y.T, test_size = 0.2, random_state = 1)
        X_train = X_train.T
        X_test = X_test.T
        Y_train = Y_train.reshape((1, len(Y_train)))
        Y_test = Y_test.reshape((1, len(Y_test)))
        return X_train, X_test, Y_train, Y_test
    elif prediction_mode:
        Y = np.array([data[1]])
        Y = np.where(Y == "M", 1, 0)
        X = np.array(data.iloc[:, 2:])
        X = normalize_data(X)
        X = X.T
        return X, Y

def update_gradiant(dW1, db1, dW2, db2, grads):
    grads['dW1'] = dW1
    grads['db1'] = db1
    grads['dW2'] = dW2
    grads['db2'] = db2
    return grads

def unpack_model_params(params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    return W1, b1, W2, b2

def plot_cost_loss(costs, losses):
    plt.plot(np.squeeze(costs), 'b', label = 'cost')
    plt.plot(np.squeeze(losses), 'r', label = 'value loss')
    plt.ylabel('cost and loss values')
    plt.xlabel('Iterations')
    plt.title('Cost (blue) and Loss (red)')
    plt.show()

def print_epoch_state(i, epoch, cost, loss, lr):
    print("Epoch {}/{} - loss : {} - val_loss : {} - learning_rate : {:.8f}".format(i + 1, epoch, "%.4f" % cost, "%.4f" % loss, lr))

def print_prediction(X, Y, params, train_or_test):
    accuracy, y = mlp.predict(X, Y, params)
    print("Accuracy on the {} set : {:.2f}".format("training" if train_or_test else "test", accuracy * 100))

def update_model_parameters(params, gradiants, lr):
    for i in range(len(params) // 2):
        params['W' + str(i + 1)] -= lr * gradiants['dW' + str(i + 1)]
        params['b' + str(i + 1)] -= lr * gradiants['db' + str(i + 1)]
    return params['W1'], params['b1'], params['W2'], params['b2']
