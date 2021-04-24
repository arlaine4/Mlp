import numpy as np
import utils
import matplotlib.pyplot as plt

def sigmoid(activation):
    """
    Sigmoid activation function
    Its called to calculate the values of the next layer
    neurons
    """
    sigmoid = 1 / (1 + np.exp(-activation))
    return s

def sigmoid_derivate(dA, z):
    """
    Same as sigmoid but this one is used during
    back propagation (backward_pass)

    We first get the sigmoid value and then
    calculate its derivate
    """
    s = 1 / (1 + np.exp(-z))
    derivate = dA * s * (1 - s)
    return derivate

def derivate_relu(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(x):
    return (np.exp(x) / np.sum(np.exp(x) + 1e-6, axis = 0))

def forward_pass(prev, W, B, activation_func, return_cached = True):
    Z = np.dot(W,prev) + B
    if activation_func == "softmax":
        activation = softmax(Z)
    elif activation_func == "relu":
        activation = np.maximum(Z, 0)
    if return_cached:
        cached = (prev, W, B, Z)
        return activation, cached
    else:
        return activation

def backward_pass(dA, AL, Y, cached, activation_func):
    A_prev, W, B, Z = cached
    m = A_prev.shape[1]
    if activation_func == "softmax":
        dZ = AL - Y
    elif activation_func == "relu":
        dZ = derivate_relu(dA, Z)
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def binary_cross_entropy(actual, predicted):
    sum_score = actual * np.log(1e-6 + predicted)
    mean_sum_score = 1.0 / np.size(actual, axis = 0) * sum_score
    return sum_score

def calculate_cost(al, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(al + 1e-6) + (1 - Y) * np.log(1 - al + 1e-6), 
            keepdims=True, axis = 1)
    cost = np.squeeze(cost)
    return cost

def predict(X, Y, params):
    W1, b1, W2, b2 = utils.unpack_model_params(params)
    A1 = forward_pass(X, W1, b1, 'relu', False)
    y = forward_pass(A1, W2, b2, 'softmax', False)
    y = np.where(y > 0.5, 1, 0)
    res = (y == Y).mean()
    return res, y

def neural_network(X, Y, X_test, Y_test, s_x, s_h, s_y, params):
    lr = 0.01
    epsilon = 1e-6
    epoch = 10000
    W1, b1, W2, b2 = utils.unpack_model_params(params)
    gradiants = {}
    costs = []
    value_losses = []
    for i in range(epoch):
        # Forward pass for hidden layers
        #   First hidden layer fed with inputs
        a1, cached1 = forward_pass(X, W1, b1, 'relu')
        #   Second hidden layer fed with values from first hidden layer
        a2, cached2 = forward_pass(a1, W2, b2, 'softmax')
        # Doing a pass with test set to later calculate loss value
        a1_t = forward_pass(X_test, W1, b1, 'relu', False)
        a2_t = forward_pass(a1_t, W2, b2, 'softmax', False)
        cost = calculate_cost(a2, Y)
        loss = calculate_cost(a2_t, Y_test)
        da2 = -(np.divide(Y, a2 + 1e-6) - np.divide(1 - Y, 1 - a2 + 1e-6))
        da1, dW2, db2 = backward_pass(da2, a2, Y, cached2, 'softmax')
        da0, dW1, db1 = backward_pass(da1, a2, Y, cached1, 'relu')
        gradiants = utils.update_gradiant(dW1, db1, dW2, db2, gradiants)
        W1, b1, W2, b2 = utils.update_model_parameters(params, gradiants, lr)
        if i > 2500 and i % 100 == 0:
            lr = (1. / (1.+ lr * epoch))
        if i % 100 == 0:
            utils.print_epoch_state(i, epoch, cost, loss, lr)
            costs.append(cost)
            value_losses.append(loss)
    plt.plot(np.squeeze(costs), 'b', label='costs')
    plt.plot(np.squeeze(value_losses), 'r', label='val_loss')
    plt.ylabel('cost')
    plt.xlabel('value per one hundred iteration')
    plt.title('Cost over iterations')
    plt.show()
    return params
