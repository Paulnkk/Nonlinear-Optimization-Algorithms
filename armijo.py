import numpy as np
from scipy import optimize as opt

"""
Armijo linesearch Algorithm
"""

def schrittweite(f, fd, xk, d, sigma, rho, gamma):

    dnorm = (np.linalg.norm(d))**2
    t = - gamma * (np.dot(fd(xk), d) / dnorm)
    while f(xk + t * d) > f(xk) + t * sigma * np.dot(fd(xk), d):
        t = rho * t

    return t
