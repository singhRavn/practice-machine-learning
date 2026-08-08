import numpy as np
from numpy import linalg
import cvxxopt
import cvxopt.solvers

def linear_kernel(x1, x2):
    return np.dot(x1, x2)