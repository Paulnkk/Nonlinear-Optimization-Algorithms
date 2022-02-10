import numpy as np
from scipy import optimize as opt
import armijo
import math



class GradientMethod(object):
    """
    Constructor Gradient-Method
    """
    def __init__ (self, f, fd, H, xk, eps):
        self.xk = xk
        self.eps = eps
        self.fd = fd
        self.f = f
        self.H = H
        return
    """
    Gradient-Method
    """

    def work (self):
        f = self.f
        xk = self.xk
        eps = self.eps
        fd = self.fd
        it = 0
        maxit = 2000
        while (np.linalg.norm(fd(xk)) > eps) and (it < maxit):
            t = armijo.schrittweite(f, fd, xk, -fd(xk), sigma = 0.02, rho = 0.5, gamma = 0.0001)
            xk = xk - t * fd(xk) 
            print("Log-Values(Gradient): ", math.log10(f(xk)))
            it += 1
        return xk, it
