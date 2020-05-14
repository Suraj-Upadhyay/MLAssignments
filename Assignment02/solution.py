# Copyright 2020 by Suraj Upadhyay
# Contact info : usuraj35@gmail.com
# This file is part of https://github.com/MrSquanchee/MLAssignments
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Import necessary libraries.
import scipy.optimize as spo
import numpy as np
import sys
import os

# Import helper librarires.
from hypothesis import hypothesis
from plot_data import plot_data
from gradient import gradient
from cost import cost

def getoptions() :
    if (len(sys.argv) != 2) :
        print("Incorrect Usage")
        print("Should use : python solution.py __datafilename__")
        exit()

    return sys.argv[1]

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
    X = np.hstack((np.ones((m, 1)), X)).reshape(m, n + 1)

    # Prepare the target vector.
    y = data[:, -1].reshape((m, 1))
    # Parameter vector : Initialized to zero
    theta = np.zeros((n + 1, 1))

    return [X, y, theta]

def main() :
    file_name = getoptions()
    X, y, theta = load_and_decorate_data(file_name)
    optimal_theta = spo.fmin_bfgs(cost, theta, gradient, args = (X, y),
                                  disp = False)
    plot_data(X, y, optimal_theta)

if __name__ == "__main__" :
    main()
