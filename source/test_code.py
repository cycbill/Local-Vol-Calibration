import numpy as np 
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt

from parameters import *

from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.stats import norm
from black_scholes_formulas import black_scholes_vanilla

#x = np.array([1, 2, 2.1, 3, 4, 5])
#y = np.array([1.0, 1.1, 1.5, 1.8, 2.8, 5.6])

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