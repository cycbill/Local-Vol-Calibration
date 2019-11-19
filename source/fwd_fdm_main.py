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

## Trade and single data
callput = 1
S = 0.6
T = 1

## Curve Construction
T_inputs = np.array([0.01, 10.0])
r_inputs = np.array([0.05, 0.05])
rf_inputs = np.array([0.0, 0.0])
r_para = RateCurve(T_inputs, r_inputs)
rf_para = RateCurve(T_inputs, rf_inputs)

## Tenor Market Data
tenor_mkt_data = TenorMarketData(S, r_para, rf_para, T)

## k inputs
imp_vol_atm = 0.2
loc_vol_inputs = np.array([0.25, 0.15])
k_inputs = np.array([-5*imp_vol_atm*np.sqrt(T), 5*imp_vol_atm*np.sqrt(T)])
K_inputs = tenor_mkt_data.fwd * np.exp(k_inputs)

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


K_outputs = tenor_mkt_data.fwd * np.exp(x_values)
payoff_outputs = init_cond.compute(x_values) * S
premium_outputs = prices * tenor_mkt_data.fwd * tenor_mkt_data.DF_r
premium_bs_vatm = black_scholes_vanilla(callput, S, K_outputs, T, tenor_mkt_data.r, tenor_mkt_data.rf, imp_vol_atm)


dual_delta_bs_analytic = black_scholes_vanilla_dual_delta(callput, S, K_outputs, T, tenor_mkt_data.r, tenor_mkt_data.rf, imp_vol_atm)
dual_gamma_bs_analytic = black_scholes_vanilla_dual_gamma(S, K_outputs, T, tenor_mkt_data.r, tenor_mkt_data.rf, imp_vol_atm)
implied_vol_guess = loc_vol_para.interpolate(x_values)


implied_vol = black_scholes_vanilla_solve_vol(callput, S, K_outputs[1:-1], T, tenor_mkt_data.r, tenor_mkt_data.rf, implied_vol_guess[1:-1], premium_outputs[1:-1])
premium_bs_implied_vol = black_scholes_vanilla(callput, S, K_outputs[1:-1], T, tenor_mkt_data.r, tenor_mkt_data.rf, implied_vol)


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



