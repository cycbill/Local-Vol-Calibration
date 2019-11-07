import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parameters import PiecewiseLinearParameter1D
from payoff import PayOffCall, PayOffPut
from option2 import CalibrationBasketVanillaOption
from initial_condition import InitialConditionFirstTenor
from fwd_fdm import FDMCrankNicolsonNeumann
from black_scholes_formulas import *
import xlwings as xw

S = 0.6
r = 0.25
T = 1.00
#loc_vol_inputs = np.repeat(0.25, 7)
loc_vol_inputs = np.array([0.25, 0.20, 0.15, 0.10, 0.15, 0.20, 0.25])
K_inputs = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
F = S * np.exp(r*T)

k_inputs = np.log(K_inputs / F)

x_min = -4.0
x_max = 1.0
J = 200
t_min = 0
t_max = T
N = 200

loc_vol_para = PiecewiseLinearParameter1D(k_inputs, loc_vol_inputs)
calib_basket = CalibrationBasketVanillaOption(S, r, T, loc_vol_para)
init_cond = InitialConditionFirstTenor()
fdm_cn = FDMCrankNicolsonNeumann(x_min, x_max, J, t_min, t_max, N, calib_basket, init_cond)

prices, x_values = fdm_cn.step_march()
plt.show()


K_outputs = F * np.exp(x_values)
payoff_outputs = calib_basket.payoff_by_logmoney(x_values) * S
premium_outputs = prices * F * np.exp(-r*T)
premium_bs = black_scholes_vanilla(S, K_outputs, T, r, 0, 0.25)

dual_delta_bs_analytic = black_scholes_vanilla_dual_delta(S, K_outputs, T, r, 0, 0.25)
dual_gamma_bs_analytic = black_scholes_vanilla_dual_gamma(S, K_outputs, T, r, 0, 0.25)
implied_vol = black_scholes_vanilla_solve_vol(S, K_outputs, T, r, 0, 0.2, premium_outputs)


## Output pde results
wb = xw.Book('LocVol Parameters.xlsx')
sht = wb.sheets['FDM_Output']

sht.range('D4').options(transpose=True).value = x_values
sht.range('E4').options(transpose=True).value = K_outputs
sht.range('F4').options(transpose=True).value = prices
sht.range('G4').options(transpose=True).value = payoff_outputs
sht.range('H4').options(transpose=True).value = premium_outputs
sht.range('I4').options(transpose=True).value = premium_bs
sht.range('M4').options(transpose=True).value = dual_delta_bs_analytic
sht.range('Q4').options(transpose=True).value = dual_gamma_bs_analytic

sht.range('B3').value = S
sht.range('B4').value = r
sht.range('B5').value = T
sht.range('B6').value = 0.25

