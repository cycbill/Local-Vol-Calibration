import numpy as np 
from scipy.optimize import minimize, LinearConstraint

def func(x):
    result = (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] + 3)**2 + (x[3] - 1)**2
    return result

def print_callback(xk):
    n += 1
    print(n, xk, func(xk))


x0 = np.array([1,2,3,4])

A = np.identity(4)
lb = np.array([-4, -4, -4, -4])
ub = np.array([4, 4, 4, 4])

lr = LinearConstraint(A, lb, ub)

n=0

res = minimize(func, x0, method='BFGS', constraints=lr, callback=print_callback)

print('Final result = ', res.x, 'Iter times = ', res.nit)