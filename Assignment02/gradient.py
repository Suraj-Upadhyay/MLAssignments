# Copyright 2020 by Suraj Upadhyay
# Contact info : usuraj35@gmail.com
# This file is part of https://github.com/MrSquanchee/MLAssignments
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Import necessary libraries.
import numpy as np

# Import helper libraries.
from hypothesis import hypothesis

def gradient(theta : np.array, X : np.array, y : np.array) -> np.array :
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
    # Calculate the derivative of the cost function.
    derivative = np.sum((hypothesis(X, theta) - y) * X, axis = 0)

    return derivative
