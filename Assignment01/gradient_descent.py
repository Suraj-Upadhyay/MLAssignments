# Copyright 2020 by Suraj Upadhyay
# Contact info : usuraj35@gmail.com
# This file is part of https://github.com/MrSquanchee/MLAssignments
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Import necessary libraries
import numpy as np

# Import helper modules.
from cost import costFunction
from gradient import gradient

def gradientDescent(X, y, alpha, theta, iter) -> list :
    m = len(X)

    costHistory = [costFunction(X, y, theta)]
    prevTheta = theta.copy()

    for i in range(iter) :
        mGradient = gradient(X, y, theta)
        theta = theta - (alpha / m) * mGradient
        cost = costFunction(X, y, theta)
        costHistory.append(cost)
        if cost > costHistory[i] :
            print('Exiting at iteration', i+1)
            return [prevTheta, costHistory]
        prevTheta = theta.copy()

    return [theta, costHistory]
