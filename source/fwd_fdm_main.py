import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlwings as xw

from parameters import PiecewiseLinearParameter1D
from rate_curve_class import RateCurve
from tenor_market_data import TenorMarketData
from initial_condition import InitialConditionFirstTenor
from fwd_fdm import FDMCrankNicolsonNeumann
from black_scholes_formulas import *


callput = 1
S = 0.6
r = 0.05
rf = 0.00
T = 1

r_para = RateCurve(r) ################################################## need to change ##############################################
tenor_mkt_data = TenorMarketData(S, r, rf, T)


F = S * np.exp(r*T)

## K inputs
#loc_vol_inputs = np.array([0.25, 0.20, 0.15, 0.10, 0.15, 0.20, 0.25])
#K_inputs = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
#k_inputs = np.log(K_inputs / F)

## k inputs
imp_vol_atm = 0.21
loc_vol_inputs = np.array([0.21, 0.21])
k_inputs = np.array([-5*imp_vol_atm*np.sqrt(T), 5*imp_vol_atm*np.sqrt(T)])
K_inputs = F * np.exp(k_inputs)

x_min = k_inputs[0]
x_max = k_inputs[-1]
J = 200
t_min = 0
t_max = T
N = 2000

loc_vol_para = PiecewiseLinearParameter1D(k_inputs, loc_vol_inputs)

init_cond = InitialConditionFirstTenor()
fdm_cn = FDMCrankNicolsonNeumann(x_min, x_max, J, t_min, t_max, N, tenor_mkt_data, loc_vol_para, init_cond)

prices, x_values = fdm_cn.step_march()
plt.show()


K_outputs = F * np.exp(x_values)
payoff_outputs = init_cond.compute(x_values) * S
premium_outputs = prices * F * np.exp(-r*T)
premium_bs_vatm = black_scholes_vanilla(callput, S, K_outputs, T, r, 0, imp_vol_atm)
'''
pde_price_interpolator = PiecewiseLinearParameter1D(K_outputs, premium_outputs)
pde_price = pde_price_interpolator.interpolate(0.83889174)

bs_price = black_scholes_vanilla(S, 0.83889174, T, r, 0, imp_vol_atm)

print(pde_price, bs_price, pde_price-bs_price)

'''
dual_delta_bs_analytic = black_scholes_vanilla_dual_delta(callput, S, K_outputs, T, r, 0, imp_vol_atm)
dual_gamma_bs_analytic = black_scholes_vanilla_dual_gamma(S, K_outputs, T, r, 0, imp_vol_atm)
implied_vol_guess = loc_vol_para.interpolate(x_values)


implied_vol = black_scholes_vanilla_solve_vol(callput, S, K_outputs[1:-1], T, r, 0, implied_vol_guess[1:-1], premium_outputs[1:-1])
premium_bs_implied_vol = black_scholes_vanilla(callput, S, K_outputs[1:-1], T, r, 0, implied_vol)


## Output pde results
wb = xw.Book('LocVol Parameters.xlsx')
sht = wb.sheets['FDM_Output']

sht.range('E4').options(transpose=True).value = x_values
sht.range('F4').options(transpose=True).value = K_outputs
sht.range('G4').options(transpose=True).value = prices
sht.range('H4').options(transpose=True).value = payoff_outputs
sht.range('I4').options(transpose=True).value = premium_outputs
sht.range('J5').options(transpose=True).value = implied_vol
sht.range('K5').options(transpose=True).value = premium_bs_implied_vol
sht.range('L4').options(transpose=True).value = premium_bs_vatm
sht.range('Q4').options(transpose=True).value = dual_delta_bs_analytic
sht.range('X4').options(transpose=True).value = dual_gamma_bs_analytic



sht.range('B3').value = S
sht.range('B4').value = r
sht.range('B5').value = T
sht.range('B6').value = 0.25

