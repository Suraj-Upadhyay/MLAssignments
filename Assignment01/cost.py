# Copyright 2020 by Suraj Upadhyay
# Contact info : usuraj35@gmail.com
# This file is part of https://github.com/MrSquanchee/MLAssignments
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Import necessary libraries.
import numpy as np

def costFunction(X : np.array, y : np.array, theta : np.array):
    """
    Return half the mean squared error.

    Input: X (Features), y (predictions), theta (parameters).
    """

    # Calculate the hypothesis/prediction.
    prediction = X @ theta
    # Calculate the error for each prediction.
    error = prediction - y
    # Errors Squared.
    error = error ** 2
    # Mean/Average of the errors squared.
    avg_error = np.sum(error) / len(X)

    return avg_error / 2
