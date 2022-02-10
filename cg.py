import numpy as np
from numpy.linalg import inv
from scipy import optimize as opt
import math
import armijo

class CGMethod(object):
    """
    Constructor CG-Method
    """
    def __init__ (self, f, fd, H, xk, eps):
        self.fd = fd
        self.f = f
        self.H = H
        self.xk = xk
        self.eps = eps
        return
    """
    CG-Method
    """
    def work (self):
        f = self.f
        fd = self.fd
        H = self.H
        xk = self.xk
        eps = self.eps
        t = 1
        it = 0
        #maxit = 10000
        dprev = - fd(xk)
        xprev = self.xk
        dk    = None

        while(np.linalg.norm(fd(xk)) > eps): #and (it < maxit):
            t = armijo.schrittweite(f, fd, xprev, dprev, sigma = 0.02, rho = 0.5, gamma = 2)
            xk = xprev + t * dprev
            normprev = (np.linalg.norm(fd(xprev)))**2
            xprev = xk
            normk = (np.linalg.norm(fd(xk)))**2
            dk = - fd(xk) + (normk / normprev)*(dprev)
            dprev = dk
            print("Log-Values(CG): ", math.log10(f(xk)))
            it += 1

        return xk, it
