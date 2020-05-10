# Copyright 2020 by Suraj Upadhyay
# Contact info : usuraj35@gmail.com
# This file is part of https://github.com/MrSquanchee/MLAssignments
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Import necessary libraries.
import numpy as np

def gradient(X, y, theta) :
    n = len(X[0])
    return np.sum((X @ theta - y) * X, axis = 0).reshape((n, 1))
