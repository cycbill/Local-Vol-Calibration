import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parameters import PiecewiseLinearParameter1D
from payoff import PayOffCall, PayOffPut
from option2 import CalibrationBasketVanillaOption
from fwd_pde import ForwardPDE
from fwd_fdm import FDMCrankNicolsonNeumann


S = 0.5
r = 0.25
loc_vol_inputs = np.repeat(0.2, 7)
k_inputs = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
T = 1.00

x_min = -1.0
x_dom = 1.0
J = 21
t_dom = T
N = 21

loc_vol_para = PiecewiseLinearParameter1D(k_inputs, loc_vol_inputs)
call_option = CalibrationBasketVanillaOption(S, r, T, loc_vol_para)
bs_pde = ForwardPDE(call_option)
fdm_euler = FDMCrankNicolsonNeumann(x_min, x_dom, J, t_dom, N, bs_pde)

prices = fdm_euler.step_march()

plt.plot(prices)
plt.show()