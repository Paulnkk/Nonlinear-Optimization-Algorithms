# Nonlinear Optimization Algorithms

During my time as Scientific Assistant at the Karlsruhe Institute of Technology (Germany) I implemented various Optimization Algorithms solving unrestricted nonlinear Problems; Gradient-Descent-Method, Newton-Method, Conjugate-Gradient-Descent-Method, BFGS-Method and a Trust-Region-Method in Python.

In addition, I implemented an Armijo linesearch.

The code is implemented in an object-oriented manner, whereby each method is implemented in a class (bfgs.py, cg.py, gradv.py, newtonm.py and tr.py) and executed via the ros_test.py script. In the script ros_test.py the Rosenbrock function was implemented, which is minimized to a given starting point x_0 with each method.
