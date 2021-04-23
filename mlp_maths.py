import numpy as np

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

def derivate_relu(dA, z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def forward_pass(prev, W, B, activation_func):
    Z = np.dot(W, prev) + B
    if activation_func == "softmax":
        activation = softmax(Z)
    elif activation_func == "relu":
        activation = np.maximum(Z, 0)
    cached = (prev, W, B, Z)
    return activation, cached

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
