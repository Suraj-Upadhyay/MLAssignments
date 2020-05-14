# Copyright 2020 by Suraj Upadhyay
# Contact info : usuraj35@gmail.com
# This file is part of https://github.com/MrSquanchee/MLAssignments
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Import necessary libraries.
import matplotlib.pyplot as plt
import numpy as np

def plot_data(X : np.array, y : np.array, theta : np.array) -> None :
    """
    Plot data for bivariate training set and the decision boundary.

    Parameters
    ----------
    X : Numpy array.
        Represents a bivariate training set.

    y : Numpy array.
        Labels for the training set.

    theta : Numpy array.
            The parameter vector. Hopefully it contains optimal parameters,
            which can represent the decision boundary sensibly.
    """

    # Plot the training set.
    positives = np.array([[X[i][1], X[i][2]] for i in range(len(X)) if y[i] == 1])

    negatives = np.array([[X[i][1], X[i][2]] for i in range(len(X)) if y[i] == 0])

    plt.scatter(positives[:, 0], positives[:, 1], color = "b")
    plt.scatter(negatives[:, 0], negatives[:, 1], marker = 'x', color = "r")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    # Plot the decision boundary.
    x = np.linspace(0, 100, 100)
    y = -(theta[0] + theta[1]*x) / theta[2]
    plt.plot(x, y, color = "#ffff00")
    plt.ylim((20,100))
    plt.xlim((20,100))
    plt.show()