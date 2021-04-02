import argparse
import sys
import pandas as pd


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

