import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parameters import PiecewiseLinearParameter1D
from payoff import PayOffCall, PayOffPut
from option2 import CalibrationBasketVanillaOption
from fwd_pde import ForwardPDE
from fwd_fdm import FDMCrankNicolsonNeumann
from black_scholes_formulas import black_scholes_vanilla


S = 0.5
r = 0.25
T = 1.00
loc_vols = np.repeat(0.25, 7)
ks = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
F = S * np.exp(r*T)

Ks =F * np.exp(ks)


x_min = -1.0
x_dom = 1.0
J = 21
t_dom = T
N = 21

loc_vol_para = PiecewiseLinearParameter1D(ks, loc_vols)
call_option = CalibrationBasketVanillaOption(S, r, T, loc_vol_para)
bs_pde = ForwardPDE(call_option)
fdm_euler = FDMCrankNicolsonNeumann(x_min, x_dom, J, t_dom, N, bs_pde)

prices, x_values = fdm_euler.step_march()

#plt.plot(x_values, prices)
plt.show()


print("Forward: ", F)
print("Strike: ", Ks[3])
print("log moneyness: ", ks[3])
price_interp_func = PiecewiseLinearParameter1D(x_values, prices)
print("premium: ", price_interp_func.interpolate(ks[3]))


# Black Scholes analytical formula
K_values = F * np.exp(x_values)
bs_price_by_strike = lambda K: black_scholes_vanilla(S, K, T, r, 0, 0.25)
bs_closing_prices = bs_price_by_strike(K_values)
plt.plot(x_values, bs_closing_prices, 'r')
plt.plot(x_values, prices, 'b--')
plt.show()