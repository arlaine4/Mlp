import utils
from sklearn.model_selection import train_test_split
import sys

if __name__ == "__main__":
    reader, diagnosis_values = utils.get_csv_infos()
    X_train, Y_train, X_test, Y_test = train_test_split(reader, diagnosis_values)
    # X_train = each stem cell train data -> Y_train = each stem cell associated diagnosis
    # X_test, Y_test -> Same but with test -> Used for testing the neural network and making predictions
    args_lst = utils.parse_args()
    exported_weights = utils.get_training_data() # Getting data from training, None if training isn't done yet
    error_msg, bool_error = utils.check_valid_args(args_lst, exported_weights)
    if bool_error:
        sys.exit(print(error_msg))
