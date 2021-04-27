import numpy as np
import utils
import sys
import mlp_maths as mlp
import pickle

if __name__ == "__main__":
    options = utils.get_arguments()
    check_options, error_msg = utils.check_arguments(options)
    sys.exit(error_msg) if check_options else 0
    if options.mode == 'learning':
		# Unpacking train data + train labels and test data + test labels
        X_train, X_test, Y_train, Y_test = utils.get_data(options.data_path)
		# Defining layers struct
        size_x, size_hidden, size_y = utils.define_layers_sizes(X_train, Y_train)
		# Initializing model
        model_params = utils.initialize_model_structure(size_x, size_hidden, size_y)
		# Neural network loop
        model_params = mlp.neural_network(X_train, Y_train, X_test, Y_test, size_x, size_hidden, size_y, model_params, options)

        utils.print_prediction(X_train, Y_train, model_params, True)
        utils.print_prediction(X_test, Y_test, model_params, False)
        try:
            with open("model_params.pkl", 'wb') as fd:
                pickle.dump(model_params, fd)
        except:
            sys.exit("Issue with model dump")
    elif options.mode == "prediction":
        try:
            with open('model_params.pkl', 'rb') as fd:
                    parameters = pickle.load(fd)
        except:
            sys.exit("Can't find model_params.pkl, please run the program in 'learning' mode first")
        X, Y = utils.get_data(options.data_path, True)
        accuracy, yhat = mlp.predict(X, Y, parameters)
        print("Accuracy for {} : {:.2f}".format(str(options.data_path), accuracy * 100))

