import numpy as np
import utils
import sys
import mlp_maths as mlp

if __name__ == "__main__":
    options = utils.get_arguments()
    check_options, error_msg = utils.check_arguments(options)
    sys.exit(error_msg) if check_options else 0
    if options.mode == 'learning':
        X_train, X_test, Y_train, Y_test = utils.get_data(options.data_path)
        size_x, size_hidden, size_y = utils.define_layers_sizes(X_train, Y_train)
        model_params = utils.initialize_model_structure(size_x, size_hidden, size_y)
        model_params = mlp.neural_network(X_train, Y_train, X_test, Y_test, size_x, size_hidden, size_y, model_params)

        utils.print_prediction(X_train, Y_train, model_params, True)
        utils.print_prediction(X_test, Y_test, model_params, False)
