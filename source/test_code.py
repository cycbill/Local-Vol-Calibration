import numpy as np 
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt

from parameters import *

S = 0.6
r = 0.05
T = 1
F = S * np.exp(r*T)

K = np.array([0.43530155, 0.54179901, 0.63528203, 0.72456344, 0.83889174])
k = np.log(K / F)

imp_vol_inputs = np.array([0.25, 0.22, 0.20, 0.19, 0.21])

imp_vol_para1 = CubicSplineParameter1D(K, imp_vol_inputs)
imp_vol_para2 = CubicSplineParameter1D(k, imp_vol_inputs)

K_grid = np.linspace(0.4, 0.9, num=51, endpoint=True)
vol_grid1 = imp_vol_para1.interpolate(K_grid)

k_grid = np.log(K_grid / F)
vol_grid2 = imp_vol_para2.interpolate(k_grid)

plt.plot(K_grid, vol_grid1, 'r')
plt.plot(K_grid, vol_grid2, 'b')
plt.show()