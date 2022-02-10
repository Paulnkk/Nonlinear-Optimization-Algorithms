import numpy as np
from numpy.linalg import inv
from scipy import optimize as opt
import math
import armijo

class BFGS(object):
    """
    Constructor BFGS
    """
    def __init__ (self, f, fd, H, xk, eps):
        self.fd = fd
        self.H = H
        self.xk = xk
        self.eps = eps
        self.f = f
        return
    """
    BFGS-Method 
    """

    def work (self):
        f = self.f
        fd = self.fd
        H = self.H
        xk = self.xk
        eps = self.eps
        """
        Initial Matrix for BFGS (Identitiy Matrix)
        """
        E =  np.array([   [1.,         0.],
                            [0.,         1.] ])
        xprev = xk
        it = 0
        maxit = 10000

        while (np.linalg.norm(fd(xk)) > eps) and (it < maxit):
            Hfd = inv(E)@fd(xprev)
            xk = xprev - Hfd
            sk = np.subtract(xk, xprev)
            yk = np.subtract(fd(xk), fd(xprev))

            b1 = (1 / np.dot(yk, sk))*(np.outer(yk, yk))
            sub1b2 = np.outer(sk, sk)
            Esk = E @ (sk)
            sub2b2 = (1 / np.dot(sk, Esk))
            sub3b2 = np.matmul(E, sub1b2)
            sub4b2 = np.matmul(sub3b2, E)
            b2 = sub2b2 * sub4b2
            E1 = np.add(E, b1)
            E = np.subtract(E1, b2)

            xprev = xk
            print("Log-Values(BFGS): ", math.log10(f(xk)))
            it += 1

        return xk, it
