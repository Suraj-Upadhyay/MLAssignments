# Copyright 2020 by Suraj Upadhyay
# Contact info : usuraj35@gmail.com
# This file is part of https://github.com/MrSquanchee/MLAssignments
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Import necessary libraries.
import numpy as np
import sys

# Helper funcition for calculating the hypothesis.
def _sigmoid_function(expression_vector : np.array) -> float :
    """
    Calculate the sigmoid function.

    Parameters
    ----------
    expression_vector : Numpy array.

    Returns
    -------
    Sigmoid of the argument expression.
    """

    try :
        exponent = np.exp(-expression_vector)
        ret = 1 / (1 + exponent)
    except (OverflowError) :
        print("Overflow error encountered in sigmoid function")
        return 0
    return ret

def hypothesis(X : np.array, theta: np.array) -> float :
    """
    Calculate the hypothesis for logistic regression.

    Parameters
    ----------
    X : Numpy array.
        Representing the data set used in training.
    
    theta : Numpu array.
            The parameter vector.
    
    Returns
    -------
    The value of the hypothesis.
    """

    # Return the sigmoid function value for
    # the product of features and their respective
    # parameters.
    return _sigmoid_function(X @ theta)

# Driver to independently run and test
# the sigmoid function.
if __name__ == "__main__" :
    num = list(map(float, input().rstrip().split()))
    num = np.array(num)
    print(_sigmoid_function(num))
