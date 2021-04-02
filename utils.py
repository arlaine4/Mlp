import pandas as pd
import argparse
import sys


def get_csv_infos():
    """

        I  - Replacing columns indexes with nicer names, corresponding to dataset documentation
        II - Replacing M (malign) and B (Benign) inside diagnosis columns with 0 and 1 to make it easier to exploit
        later; 0 is for benign stem cell and 1 is for malign stem cell

        :return: pandas.core.frame.DataFrame -> (569 rows && 30 columns)

    """
    lst_names = ['ID', 'Diagnosis', 'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness',
                 'Concavity', 'Concave points', 'Symmetry', 'Fractal Dimension', 'Radius Mean',
                 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean',
                 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean', 'Radius Worst',
                 'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst', 'Compactness Worst',
                 'Concavity Worst',
                 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst']
    try:
        reader = pd.read_csv("data.csv", names=lst_names)
    except:
        print("Error loading dataset.")
        sys.exit(0)
    del reader['ID']
    reader, diagnosis_values = replace_m_b_with_binary(reader)
    return reader, diagnosis_values


def replace_m_b_with_binary(reader):
    """
        Remove a column inside dataset
        :param reader: panda.core.frame.dataFrame
        :return: panda.core.frame.dataFrame
    """
    diagnosis = list(range(len(reader['Diagnosis'])))
    for index, value in enumerate(reader['Diagnosis']):
        diagnosis[index] = 0 if value == 'B' else 1
    # Deleting Diagnosis row because we don't want to see it otherwise it's cheating ^^
    del reader['Diagnosis']
    return reader, diagnosis


# TODO
def get_training_data():
    """
        Get data from neural network training -> Used later for predictions
    :return: # TODO
    """
    # Won't return this later
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='Argument to run the training mode')
    parser.add_argument('-p', '--predict', action='store_true', help='Argument to run predictions')
    options = parser.parse_args()
    return options


def check_valid_args(args, exported_data):
    """
        Basic argument checker, give accurate error message
        :param args: argparse.ArgumentParser()
        :param exported_data: DONT KNOW YET
        :return: str, bool
    """
    if args.train and args.predict:
        return "Can't run the program with both Training AND Predict mode, please chose only one.", True
    elif args.predict and exported_data is None:
        return "Can't run the program with Predict mode without training the network before.", True
    elif not args.predict and not args.train:
        return "Missing mode, run main with either -t (training) or -p (predict).", True
    else:
        return "OK", False
