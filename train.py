'''
I] - Create data struct
    1) - class neuron, class layer, class mlp
    2) - class mlp has a list of layers
    3) - class layer has a list of neurons
    4) - neuron instance contains weight and activation value
II] -
'''
import utils
import numpy as np

class Mlp():
    def __init__(self):
        self.nb_neurons = [30, 20, 20, 1]  # Number of neurons for each layer
        self.weights = utils.create_matrix_list(self.nb_neurons[0:3])  # create default Weights matrix list
        # above at end line : size - 1 because there is no weights for the output layer
        self.xavier_init()  # Filling weights matrices with 'random' values
        self.activations = utils.create_matrix_list(self.nb_neurons, True)
        self.alpha = 0.01  # Learning rate

    def xavier_init(self):
        """
            Weights initialization to reduce chances
            of running into the gradiant problems
                            +
            help converge to least error faster
        """
        for i in range(3):
            self.weights[i] = np.random.rand(self.nb_neurons[i], self.nb_neurons[i + 1]) \
                * np.sqrt(1 / (self.nb_neurons[i] + self.nb_neurons[i + 1]))


def main_training(X_train, Y_train):
    network = Mlp()
