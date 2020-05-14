# Copyright 2020 by Suraj Upadhyay
# Contact info : usuraj35@gmail.com
# This file is part of https://github.com/MrSquanchee/MLAssignments
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Import necessary libraries.
import numpy as np

# Import helper libraries.
from hypothesis import hypothesis

def cost(theta : np.array, X : np.array, y : np.array) -> float:
    """
    Returns the cost for logistic regression.

    Parameters
    ----------
    X : Numpy array.
        Training set.

    y : Numpy array.
        Vector representing target variables.

    theta : Numpy array.
            Parameter vector.

    Returns
    -------
    The cost, for logistic regression, given the training set, target
    variable and the parameter vector.
    """

    # Number of training examples.
    m = len(X)

    # Cost of the hypothesis when y = 1.
    cost = np.sum(np.multiply(y, np.log(hypothesis(X, theta))))

    # Suppress divide by zero warnings,
    # which occur due to zero values in np.log funcition.
    # This happens when hypothesis for a training example
    # evaluates to zero.
    with np.errstate(divide = 'ignore') :
        # Suppress the invalid value encountered warnings,
        # which occur when one of the operands of np.multiply
        # is np.nan. This happens only when np.log has encountered
        # a zero.
        with np.errstate(invalid = 'ignore') :
            # Cost of the hypothesis when y = 0.
            cost += np.sum(np.multiply(1 - y, np.log(1 - hypothesis(X, theta))))

    cost *= (-1 / m)

    return cost