import utils
import train

import sys
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    args = utils.parse_args()
    training_data = utils.import_training_data()  # Import training data, used for prediction
    error_msg, bool_args = utils.check_valid_args(args, training_data)
    if bool_args:
        sys.exit(print(error_msg))
    reader, diagnosis = utils.read_data_from_csv()
    X_train, Y_train, X_test, Y_test = utils.rework_data(reader, diagnosis)
    if args.train:
        train.main_training(X_train, Y_train)
    # X_train -> stem cells data used for training || Y_train -> stem cells diagnosis
    # X and Y_test only used for feedforward pass inside prediction program
