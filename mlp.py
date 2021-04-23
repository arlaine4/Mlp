import numpy as np
import utils
import sys

if __name__ == "__main__":
    options = utils.get_arguments()
    check_options, error_msg = utils.check_arguments(options)
    sys.exit(error_msg) if check_options else 0
    if options.mode == 'learning':
        X_train, X_test, Y_train, Y_test = utils.get_data(options.data_path)
        size_x, size_hidden, size_y = utils.define_layers_sizes(X_train, Y_train)
        model_parameters = utils.initialize_model_structure(size_x, size_hidden, size_y)
