import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parameters import PiecewiseLinearParameter1D
from payoff import PayOffCall, PayOffPut
from option2 import CalibrationBasketVanillaOption
from fwd_pde import ForwardPDE
from fwd_fdm import FDMCrankNicolsonNeumann


S = 0.6
r = 0.25
T = 1.00
loc_vol_inputs = np.repeat(0.25, 7)
K_inputs = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
F = S * np.exp(r*T)

k_inputs = np.log(K_inputs / F)


x_min = -1.0
x_dom = 1.0
J = 21
t_dom = T
N = 21

loc_vol_para = PiecewiseLinearParameter1D(k_inputs, loc_vol_inputs)
call_option = CalibrationBasketVanillaOption(S, r, T, loc_vol_para)
bs_pde = ForwardPDE(call_option)
fdm_euler = FDMCrankNicolsonNeumann(x_min, x_dom, J, t_dom, N, bs_pde)

prices, x_values = fdm_euler.step_march()

plt.plot(x_values, prices)
plt.show()

print("Forward: ", F)
print("Strike: ", K_inputs[3])
print("log moneyness: ", k_inputs[3])
price_interp_func = PiecewiseLinearParameter1D(x_values, prices)
print("premium: ", price_interp_func.interpolate(k_inputs[3]))