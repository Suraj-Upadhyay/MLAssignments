# Copyright 2020 by Suraj Upadhyay
# Contact info : usuraj35@gmail.com
# This file is part of https://github.com/MrSquanchee/MLAssignments
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

# Import necessary libraries.
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import helper modules.
from cost import costFunction
from gradient_descent import gradientDescent

if __name__ == "__main__" :

    if len(sys.argv) != 4 :
        print("Correct use : gradientDescent.py datafilename iteration alpha")
        exit()

    fileName = sys.argv[1]
    iter = int(sys.argv[2])
    alpha = float(sys.argv[3])

    cwd = os.getcwd()
    data = np.loadtxt(cwd+'/'+fileName)

    m = len(data)
    n = len(data[0])
    n -= 1
    X = data[:, :n]
    y = data[:, -1].reshape((m, 1))
    X0 = np.ones((m, 1))
    X = np.hstack((X0, X))
    theta = np.zeros((n+1, 1), dtype=np.float128)
    X.astype(np.float128)
    y.astype(np.float128)

    theta , costHistory = gradientDescent(X, y, alpha, theta, iter)

    print(len(costHistory))
    print(theta, '\n', costFunction(X,y,theta))

    plt.plot(costHistory)
    plt.show()
