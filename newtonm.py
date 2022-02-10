import numpy as np
from numpy.linalg import inv
from scipy import optimize as opt
import math


class NewtonMethod(object):
    """
    Constructor Newton-Method
    """
    def __init__ (self, f, fd, H, xk, eps):
        self.fd = fd
        self.H = H
        self.xk = xk
        self.eps = eps
        self.f = f
        return
    """
    Newton-Method
    """
    def work (self):
        f = self.f
        fd = self.fd
        H = self.H
        xk = self.xk
        eps = self.eps
        it = 0
        #maxit = 10000

        while (np.linalg.norm(fd(xk)) > eps): #and (it < maxit):
            Hfd = inv(H(xk))@(fd(xk))
            xk = xk - Hfd
            it += 1
            print("Log-Values(Newton): ", math.log10(f(xk)))

        return xk, it
