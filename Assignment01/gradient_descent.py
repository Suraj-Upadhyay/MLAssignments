# Copyright 2020 by Suraj Upadhyay
# Contact info : usuraj35@gmail.com
# This file is part of https://github.com/MrSquanchee/MLAssignments
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Import necessary libraries
import numpy as np

# Import helper modules.
from cost import cost_function
from gradient import gradient

def gradient_descent(X : np.array, y : np.array, alpha : float,
                    theta : np.array, iter : int) -> list :
    """
    Calculate and return the parameter vector minimizing the cost function.

    Input: X (Data sets), y (Target variable), alpha (Step),
           theta (Parameter vector), iter (Number of iterations)

    Output: Returns a list containing the parameter vector and the cost history

    Expects a pre-initialized parameter vector theta, zero vector by default.
    This function tries to minimize the cost function w.r.t. the parameter
    vector. Performance of Gradient Descent depends on the number of iterations
    and the step constant.
    """
    # Number of training examples.
    m = len(X)

    # Maintain a record for costs across iterations.
    # Used for debugging the model.
    cost_history = [cost_function(X, y, theta)]

    # Maintain a copy of the previous value of theta.
    # Used for reference in-case the algorithm diverges.
    prev_theta = theta.copy()

    # The Gradient Descent Algorithm for Linear Regression.
    for i in range(iter) :
        derivative = gradient(X, y, theta)
        theta = theta - (alpha / m) * derivative

        # Calculate the cost with new set of parameters.
        cost = cost_function(X, y, theta)
        cost_history.append(cost)

        # Check for divergence.
        if cost > cost_history[i] :
            print('Exiting at iteration', i+1)
            # Return the previous set of parameters and
            # the cost function for scrutiny.
            return [prev_theta, cost_history[:-1]]

        prev_theta = theta.copy()

    return [theta, cost_history]
