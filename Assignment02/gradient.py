# Copyright 2020 by Suraj Upadhyay
# Contact info : usuraj35@gmail.com
# This file is part of https://github.com/MrSquanchee/MLAssignments
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Import necessary libraries.
import numpy as np

def gradient(X : np.array, y : np.array, theta : np.array) -> np.array :
    """
    Calculate the derivative of the cost function w.r.t. the parameter vector.

    Parameters
    ----------
    X : Numpy array.
        Data set.
    y : Numpy array.
        Target variable.
    theta : Numpy array.
            Parameter vector.

    Returns
    -------
    A vector of partial derivative of the cost function w.r.t. every parameter in theta.
    """

    # Number of features.
    n = len(X[0])

    # Calculate the derivative of the cost function.
    derivative = np.sum((X @ theta - y) * X, axis = 0)

    return derivative.reshape((n, 1)) # Return in a form expected.
