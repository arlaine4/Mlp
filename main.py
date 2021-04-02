import sys
import utils
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = utils.parse_args()
    training_data = utils.import_training_data()  # Import training data, used for prediction
    error_msg, bool_args = utils.check_valid_args(parser, training_data)
    if bool_args:
        sys.exit(print(error_msg))
    reader, diagnosis = utils.read_data_from_csv()
    X_train, Y_train, X_test, Y_test = train_test_split(reader, diagnosis)
    # X_train -> stem cells data used for training || Y_train -> stem cells diagnosis
    # X and Y_test only used for feedforward pass inside prediction program
