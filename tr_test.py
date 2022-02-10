from trgv import GradientTR
from dogleg import Dogleg
import numpy as np
import time

data = np.load("LRData.npz")
    X = data['X']
    y = data['y']
    X_t = data['X_test']
    y_t = data['y_test']

    n = np.shape(X)[1]
    m = np.shape(X)[0]
    m_test = np.shape(X_t)[0]
    lam = 1e-5

def f(beta, X, y, lam, m):
        l = np.zeros(m)
        grad = np.zeros(m)
        for i in range(m):
            e = np.exp(-y[i]*beta.dot(X[i,:]))
            l[i] = np.log(1+e)
        grad = np.mean(l)+lam*beta.dot(beta)
        return grad"








class TR(object):
    """
    TR-Method taken from Grundzuege der NLO von O.Stein (Algorithm 2.8)
    """
    def trustRegion(self, fo, tr_solver,print_steps=False,eps=0.0001, maxRad=1.0,startRad=0.5,eta=0.25):
        t0 = time.time()
        k=0
        x0 = fo.x0
        f = fo.f
        fd = fo.fd
        H = fo.H
        x = fo.x0
    """
    Select Solver for TR-Subproblem
    """
        t = startRad
        while(np.linalg.norm(fd(x)) > eps):
            solver = None
            if(tr_solver == 'grad'):
                solver = GradientTR(fd(x),None,radius=t)
            elif(tr_solver == 'dogleg'):
                solver = Dogleg(fd(x), H(x), radius=t)
            else:
                print("ERROR, NO SUBPROBLEM SELECTED. EXITING")
                exit()

            d = solver.work()

            r = 0

            r = (f(x) - f(x + d))/(solver.m(np.zeros(x0.shape[0]),f(x), fd(x),H(x)) - solver.m(d,f(x), fd(x),H(x)))




            if(r < 0.25):
                t = 0.25 * np.linalg.norm(d)
            else:
                if r > 0.75 and np.isclose(np.linalg.norm(d), t, eps):
                    t = min(2*t, maxRad)
                else:
                    t = t
            if r > eta:
                x = x + d
            else:
                x = x
            if(print_steps):
                print("Now at x =",x, "where f(x) =",f(x))

            k = k + 1
            t1 = time.time()
        return x, k, t1 - t0


"""
Create function for TR-Method
"""
class Function(object):
    x0 = None
    f = None
    fd = None
    H = None

    def __init__(self, x0, f, fd, H):
        self.x0 = x0
        self.f = f
        self.fd = fd
        self.H = H

    """
    Rosenbrock-function
    """
    def testFunctionRosenbrock():
        f = lambda xy: (10*(xy[0] - xy[1]**2))**2 + (1-xy[0])**2
        fd = lambda xy: np.array([202.*xy[0] - 200*xy[1]**2 - 2, -400*xy[1]*(xy[0] - xy[1]**2)])
        H = lambda xy: np.array([   [202.,         -400.*xy[1]                              ],
                                    [-400.*xy[1],   800.*xy[1]**2 - 400.*(xy[0] - xy[1]**2) ]
                                    ])
        x0 = np.array([-1.2,1])
        return Function(x0,f,fd,H)

"""
Print results
"""

print("STARTING TESTS")
a = TR()

print("-----Testing Rosenbrock function-----")
f = Function.testFunctionRosenbrock()
s_dg, k_dg, t_dg = a.trustRegion(f,'dogleg')
s_gr, k_gr, t_gr = a.trustRegion(f,'grad')
print("Trust region result (STTCGTR): ",s_dg, "- Required steps:", k_dg, "- Time:",t_dg)
print("Trust region result (Grad): ",s_gr, "- Required steps:", k_gr, "- Time",t_gr)
assert np.allclose(s_dg, np.array([1.,1.]), rtol=1e-04, atol=1e-05)
assert np.allclose(s_gr, np.array([1.,1.]), rtol=1e-04, atol=1e-05)
print(" + Pass")
