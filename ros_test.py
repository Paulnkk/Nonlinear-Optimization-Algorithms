from gradv import GradientMethod
import numpy as np
import time
from newtonm import NewtonMethod
from numpy.linalg import inv
from cg import CGMethod
from bfgs import BFGS
import armijo



class Rosenb:

    """
    Select Solver (Gradient-Method, Newton-Method, CG-Method, BFGS-Method)
    """


    def algo(self, fo, alsolver):
        x0 = fo.x0
        f = fo.f
        fd = fo.fd
        H = fo.H
        x = fo.x0
        eps = 0.0001 
        it = None
        while(np.linalg.norm(fd(x)) > eps):
            solver = None
            if(alsolver == 'gradv'):
                #Nutze Gradientenverfahren Loesung
                solver = GradientMethod(f, fd, H, x, eps)
            elif(alsolver == 'newtonm'):
                solver = NewtonMethod(f, fd, H, x, eps)
            elif(alsolver == 'cg'):
                solver = CGMethod(f, fd, H, x, eps)
            elif(alsolver == 'bfgs'):
                solver = BFGS(f, fd, H, x, eps)
            else:
                print("ERROR, NO SOLVER SELECTED. EXITING")
                exit()

            x, it = solver.work()

        return x, it

"""
Create Function that should be minimized
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
    Rosenbrock function
    """
    def testFunctionRosenbrock():
        f = lambda xy: (10*(xy[0] - xy[1]**2))**2 + (1-xy[0])**2
        fd = lambda xy: np.array([202.*xy[0] - 200*xy[1]**2 - 2, -400*xy[1]*(xy[0] - xy[1]**2)])
        H = lambda xy: np.array([   [202.,         -400.*xy[1]                              ],
                                    [-400.*xy[1],   800.*xy[1]**2 - 400.*(xy[0] - xy[1]**2) ]
                                ])
        x0 = np.array([-2.,2.])
        return Function(x0,f,fd,H)

    """
    Simple quadratic function ((x,y) -> (x + 3)^2 + y^2)
    """
    def testFunction2():
        f = lambda xy: (xy[0] + 3)**2 + xy[1]**2
        fd = lambda xy: np.array([2*xy[0] + 6, 2*xy[1]])
        H = lambda xy: np.array([[2., 0.],[0., 2.]])
        x0 = np.array([3.,0.])
        return Function(x0,f,fd,H)


print("STARTING TESTS")
a = Rosenb()
"""
print("-----Testing quadratic(2) function-----")
f = Function.testFunction2()
x_gv, it_gv = a.algo(f,'gradv')
x_nm, it_nm = a.algo(f, 'newtonm')
x_cg, it_cg = a.algo(f, 'cg')
#x_bfgs, it_bfgs = a.algo(f, 'bfgs')
print("Result (Grad): ",x_gv, " steps: ", it_gv)
print("Result (Newton): ",x_nm, " steps: ", it_nm)
print("Result (CG): ",x_cg, "steps: ", it_cg)
#print("Result (BFGS): ",x_bfgs, "steps: ", it_bfgs)
assert np.allclose(x_gv, np.array([-3.,0.]), rtol=1e-04, atol=1e-05)
assert np.allclose(x_nm, np.array([-3.,0.]), rtol=1e-04, atol=1e-05)
assert np.allclose(x_cg, np.array([-3.,0.]), rtol=1e-04, atol=1e-05)
#assert np.allclose(x_bfgs, np.array([-3.,0.]), rtol=1e-04, atol=1e-05)
print(" + Pass")
"""

print("-----Testing Rosenbrock function-----")
f = Function.testFunctionRosenbrock()
x_gv, it_gv = a.algo(f,'gradv')
x_nm, it_nm = a.algo(f, 'newtonm')
x_cg, it_cg = a.algo(f, 'cg')
x_bfgs, it_bfgs = a.algo(f, 'bfgs')
print("Result (Grad): ",x_gv, " steps: ", it_gv)
print("Result (Newton): ",x_nm, " steps: ", it_nm)
print("Result (CG): ",x_cg, "steps: ", it_cg)
print("Result (BFGS): ",x_bfgs, "steps: ", it_bfgs)
assert np.allclose(x_gv, np.array([1.,1.]), rtol=1e-04, atol=1e-05)
assert np.allclose(x_nm, np.array([1.,1.]), rtol=1e-04, atol=1e-05)
assert np.allclose(x_cg, np.array([1.,1.]), rtol=1e-04, atol=1e-05)
assert np.allclose(x_bfgs, np.array([1.,1.]), rtol=1e-04, atol=1e-05)
print(" + Pass")
