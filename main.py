import sys
import utils

if __name__ == "__main__":
    parser = utils.parse_args()
    training_data = utils.import_training_data()
    error_msg, bool_args = utils.check_valid_args(parser, training_data)
    if bool_args:
        sys.exit(print(error_msg))
    reader = utils.read_data_from_csv()
