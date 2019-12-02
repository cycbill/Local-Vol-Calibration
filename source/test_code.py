import numpy as np 
from scipy.optimize import minimize, LinearConstraint
from scipy import interpolate
import matplotlib.pyplot as plt

from parameters import *

from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.stats import norm
from black_scholes_formulas import black_scholes_vanilla

from scipy.interpolate import CubicSpline

from cubic_spline_class import CubicSpline_LinearExtrp
from initial_condition import InitialConditionOtherTenors


x = np.linspace(0, 1, 10, endpoint=True)
y = norm.cdf(x)





left_deriv = (y[1] - y[0]) / (x[1] - x[0])
right_deriv = (y[-1] - y[-2]) / (x[-1] - x[-2])


#f = interpolate.interp1d(x, y, kind = 'cubic', fill_value='extrapolate')
f1 = CubicSpline(x, y, bc_type='natural', extrapolate=True)
f2 = CubicSpline(x, y, bc_type=((1, left_deriv), (1, right_deriv)), extrapolate=True)
f3 = CubicSpline_LinearExtrp(x, y)
f4 = InitialConditionOtherTenors(x, y)



xnew = np.linspace(-1, 2, 50, endpoint=True)
ynew1 = f1(xnew)
ynew2 = f2(xnew)
ynew3 = f3.interpolate(xnew)
ynew4 = f4.compute(xnew)

#plt.plot(x, y, 'o', xnew, ynew1, 'r-', xnew, ynew3, 'b-', xnew, ynew4, 'g-')
plt.plot(x, y, 'o', xnew, ynew4, 'g-')
plt.show()

'''
x = np.linspace(0.001, 100, 100, endpoint=True)
k = 50
v = 0.1
y = black_scholes_vanilla(1, x, k, 1, 0.03, 0, 0.1)

cs = CubicSpline(x, y)
pchip = PchipInterpolator(x, y)

xs = np.linspace(0.0001, 101, 100, endpoint=True)

y_cs = cs(xs)
y_pchip = pchip(xs)

dy_cs = np.diff(y_cs)
dy_pchip = np.diff(y_pchip)

ddy_cs = np.diff(dy_cs)
ddy_pchip = np.diff(dy_pchip)


plt.plot(xs, y_cs, 'r', label='Cubic Spline')
plt.plot(xs, y_pchip, 'b', label='PCHIP')
plt.plot(x, y, 'g.')
plt.legend(loc='upper left')
plt.title('Ys')
plt.show()

plt.plot(xs[1:], dy_cs, 'r')
plt.plot(xs[1:], dy_pchip, 'b')
plt.title('dY')
plt.show()

plt.plot(xs[2:], ddy_cs, 'r')
plt.plot(xs[2:], ddy_pchip, 'b')
plt.title('ddY')
plt.show()
'''