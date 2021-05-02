Multilayer Perceptron implementation by hand.
The goal of this project is to make prediction on breast cancer stem cells.
Look at wdbc.names for informations about the dataset

You first need to run:
python3 mlp.py data.csv (or any dataset generated) learning
-> this will train the Neural Network
--> You can a few params :
	-erl for early stopping (bases on the gradiant descent with loss values
	-plt to plot loss and cost after training

after the training you can run the program with prediction param:
python3 mlp.py data.csv (or any dataset generated) prediction 
