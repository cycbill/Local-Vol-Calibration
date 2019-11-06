import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parameters import PiecewiseLinearParameter1D
from payoff import PayOffCall, PayOffPut
from option2 import CalibrationBasketVanillaOption
from initial_condition import InitialConditionFirstTenor
from fwd_fdm import FDMCrankNicolsonNeumann
from black_scholes_formulas import black_scholes_vanilla


S = 0.6
r = 0.25
T = 1.00
loc_vol_inputs = np.repeat(0.25, 7)
K_inputs = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
F = S * np.exp(r*T)

k_inputs = np.log(K_inputs / F)

x_min = -3.0
x_max = 2.0
J = 200
t_min = 0
t_max = T
N = 200

loc_vol_para = PiecewiseLinearParameter1D(k_inputs, loc_vol_inputs)
calib_basket = CalibrationBasketVanillaOption(S, r, T, loc_vol_para)
init_cond = InitialConditionFirstTenor()
fdm_cn = FDMCrankNicolsonNeumann(x_min, x_max, J, t_min, t_max, N, calib_basket, init_cond)

prices, x_values = fdm_cn.step_march()

#plt.plot(x_values, prices)
plt.show()

print("Forward: ", F)
print("Strike: ", K_inputs[3])
print("log moneyness: ", k_inputs[3])
price_interp_func = PiecewiseLinearParameter1D(x_values, prices)
print("premium: ", price_interp_func.interpolate(k_inputs[3]))


K_outputs = F * np.exp(x_values)
payoff_outputs = calib_basket.payoff_by_logmoney(x_values) * S
premium_outputs = prices * F * np.exp(-r*T)
plt.plot(K_outputs, payoff_outputs, 'g')
plt.plot(K_outputs, premium_outputs, 'r')
plt.title('payoff and premium')
plt.show()

premium_bs = black_scholes_vanilla(S, K_outputs, T, r, 0, 0.25)
plt.plot(K_outputs, premium_bs - premium_outputs, 'g')
#plt.plot(K_outputs, premium_outputs, 'r')
plt.title('pde premium vs bs premium')
plt.show()

gamma = (premium_outputs[0:-3] - 2 * premium_outputs[1:-2] + premium_outputs[2:-1]) / ((K_outputs[2:-1] - K_outputs[0:-3]) / 2) ** 2 * K_outputs[1:-2]
plt.plot(x_values[1:-2], gamma)
plt.title('gamma')
plt.show()