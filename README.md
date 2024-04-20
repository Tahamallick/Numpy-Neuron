**Simple Neural Network from Scratch with NumPy**
This repository implements a basic neural network for binary classification from scratch using NumPy and mathematical functions.

**Features**
*Single hidden layer network with sigmoid activation function
*Implements gradient descent for training
*Calculates and prints loss (log loss) during training
*Allows early stopping based on a user-defined loss threshold

main.py takes the following arguments:
*--data_path:
Path to a CSV file containing training data. The file should have columns named 'age' and 'affordability' for features and a column named 'label' for labels (0 or 1).
*--epochs:
Number of epochs to train the network (default: 1000).
*--loss_threshold:
Minimum acceptable loss value for early stopping (default: 0.01).

**Output**
The script will print the following during training:
Epoch number
Current weights and bias values
Current loss value
After training, it will print the final weights and bias values.

**Dependencies**
*NumPy

**Disclaimer**
This is a basic implementation for educational purposes. Real-world neural networks can be much more complex and require additional techniques for optimal performance.
 
