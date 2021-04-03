import argparse
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing


def rework_data(data, diagnosis):
    X_train, X_test = separate_data(data)
    Y_train = diagnosis[400::]
    Y_test = diagnosis[400::]
    return X_train, Y_train, X_test, Y_test

def separate_data(data):
    train = data.iloc[0:400]
    true_train = []
    for i in range(len(train)):
        vec = []
        for j in range(len(train.iloc[i])):
            vec.append(train.iloc[i][j])
        vec = scale_input_data(vec)
        true_train.append(vec)
    test = data.iloc[400:]
    true_test = []
    for i in range(len(test)):
        vec = []
        for j in range(len(test.iloc[i])):
            vec.append(test.iloc[i][j])
        vec = scale_input_data(vec)
        true_test.append(vec)
    return true_train, true_test


def scale_input_data(data):
    input_a = np.array(data)
    input_a = input_a.reshape(-1, 1)
    mean_all = preprocessing.PowerTransformer()
    mean_all = mean_all.fit(input_a)
    mean_all = mean_all.transform(input_a)
    mean_all = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    new_mean_all = mean_all.fit_transform(input_a)
    last_values = []
    for i in range(len(new_mean_all)):
        last_values.append(float(new_mean_all[i]))
    return last_values


def create_matrix_list(nb_neurons, mod=False):
    """
        Create a list of numpy.zeros arrays
        -> Used as weights and activations matrices default
        builder
    :param nb_neurons -> list of ints:
    :param mod -> bool -> False when we create weight matrix, true for
        activations values matrix
    :return: list(np.zeros)
    """
    lst_matrix = [None for i in range(len(nb_neurons))]
    # nb_neurons[i] is the number of weights from the previous layer
    # and nb_neurons[i + 1] is the number of weights inside the next layer
    if not mod:
        for i in range(len(nb_neurons) - 1):
            lst_matrix[i] = np.zeros((nb_neurons[i], nb_neurons[i + 1]))
    elif mod:
        for i in range(len(nb_neurons)):
            lst_matrix[i] = np.zeros(nb_neurons[i])
    return lst_matrix


def read_data_from_csv():
    lst_names = ['ID', 'Diagnosis', 'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness',
                 'Concavity', 'Concave points', 'Symmetry', 'Fractal Dimension', 'Radius Mean',
                 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean',
                 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean', 'Radius Worst',
                 'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst', 'Compactness Worst', 'Concavity Worst',
                 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst']
    try:
        reader = pd.read_csv("data.csv", names=lst_names)
    except:
        print("Error loading dataset.")
        sys.exit(0)
    del reader['ID']
    reader, diagnosis = get_diagnosis_with_reader_change(reader)
    return reader, diagnosis


def get_diagnosis_with_reader_change(reader):
    """
        Little reader reformat, deleting diagnosis column and saving it a binary number instead of B or M
        -> 0 is for a benign stem cell and 1 is for a malign stem cell
    :param reader: panda.core.frame.dataFrame
    :return: panda.core.frame.dataFrame, list
    """
    diagnosis = list(range(len(reader['Diagnosis'])))
    for i, value in enumerate(reader['Diagnosis']):
        diagnosis[i] = 0 if value == 'B' else 1
    del reader['Diagnosis']
    return reader, diagnosis


def parse_args():
    """
        Basic argument Parser
    :return: argparse.ArgumentParser()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='training mode')
    parser.add_argument('-p', '--predict', action='store_true', help='predict mode')
    options = parser.parse_args()
    return options

# This method won't be done until the training program is working
def import_training_data():
    """
        This method allow us to gather weight and activation values from
        training
        -> used for predictions
    :return: not decided yet
    """
    # TODO
    return None


def check_valid_args(args, data):
    if data is None and args.predict and not args.train:
        return "Can't run Predict mode without training the model before.", True
    elif args.predict and args.train:
        return "Can't run the program with both Training AND Predict mode.", True
    elif not args.predict and not args.train:
        return "Please use -t (train) or -p (predict) to run the program.", True
    else:
        return "", False

