# Copyright 2020 by Suraj Upadhyay
# Contact info : usuraj35@gmail.com
# This file is part of https://github.com/MrSquanchee/MLAssignments
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Import necessary libraries.
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import helper modules.
from cost import cost_function
from gradient_descent import gradient_descent

def getoptions() :
    """
    Get the command line arguments.
    """
    if len(sys.argv) != 4 :
        print("Correct use : solution.py `datafilename` `iteration` `alpha`")
        exit()

    file_name = sys.argv[1]
    iter = int(sys.argv[2])
    alpha = float(sys.argv[3])

    return [file_name, iter, alpha]

def load_and_decorate_data(file_name) :
    """
    Load the data and convert them into appropriate expected format.

    Input: file_name (String).
    Output: X (Data set), y (Target variable),
            theta (parameter vector initialized with zero).
    Expects the data file to be in the current working directory.
    """

    # Get the path of the current working directory.
    cwd = os.getcwd()
    data = np.loadtxt(cwd+'/'+file_name)

    # Number of training sets.
    m = len(data)
    # Number of features.
    n = len(data[0]) - 1

    # Prepare the feature vector.
    X = data[:, :n]
    X = np.hstack((np.ones((m, 1)), X))

    # Prepare the target vector.
    y = data[:, -1].reshape((m, 1))
    # Parameter vector : Initialized to zero
    theta = np.zeros((n + 1, 1))

    # Cast all the values to appropriately big and precise data type.
    X.astype(np.float128)
    y.astype(np.float128)
    theta.astype(np.float128)

    return [X, y, theta]

def main() :
    """
    The driver function for Linear Regression with Gradient Descent.

    Loads the data and prepares it into appripriate format for the rest
    of the program.
    Runs the Gradient Descent for the data loaded.
    Prints and plots the run information.
    """
    # Load data in an appropriate fromat for Gradient Descent.
    file_name, iter, alpha = getoptions()
    X, y, theta = load_and_decorate_data(file_name)

    # Get the optimal parameter vector and
    # the history of Gradient Descent.
    theta , cost_history = gradient_descent(X, y, alpha, theta, iter)

    # Print the algorithm data.
    print(len(cost_history))
    print(theta, '\n', cost_function(X, y, theta))

    # Plot run history of Gradient Descent.
    plt.plot(cost_history)
    plt.show()

if __name__ == "__main__" :
    main()
