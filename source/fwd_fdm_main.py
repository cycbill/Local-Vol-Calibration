import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlwings as xw

from parameters import PiecewiseLinearParameter1D
from tenor_market_data import TenorMarketData
from initial_condition import InitialConditionFirstTenor
from fwd_fdm import FDMCrankNicolsonNeumann
from black_scholes_formulas import *


S = 0.6
r = 0.25
T = 1
F = S * np.exp(r*T)

## K inputs
#loc_vol_inputs = np.array([0.25, 0.20, 0.15, 0.10, 0.15, 0.20, 0.25])
#K_inputs = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
#k_inputs = np.log(K_inputs / F)

## k inputs
imp_vol_atm = 0.25
loc_vol_inputs = np.array([0.25, 0.15])
k_inputs = np.array([-5*imp_vol_atm*np.sqrt(T), 3*imp_vol_atm*np.sqrt(T)])
K_inputs = F * np.exp(k_inputs)

x_min = k_inputs[0]
x_max = k_inputs[-1]
J = 200
t_min = 0
t_max = T
N = 2000

loc_vol_para = PiecewiseLinearParameter1D(k_inputs, loc_vol_inputs)
tenor_mkt_data = TenorMarketData(S, r, T)
init_cond = InitialConditionFirstTenor()
fdm_cn = FDMCrankNicolsonNeumann(x_min, x_max, J, t_min, t_max, N, tenor_mkt_data, loc_vol_para, init_cond)

prices, x_values = fdm_cn.step_march()
plt.show()


K_outputs = F * np.exp(x_values)
payoff_outputs = init_cond.compute(x_values) * S
premium_outputs = prices * F * np.exp(-r*T)
premium_bs_vatm = black_scholes_vanilla(S, K_outputs, T, r, 0, imp_vol_atm)
dual_delta_bs_analytic = black_scholes_vanilla_dual_delta(S, K_outputs, T, r, 0, imp_vol_atm)
dual_gamma_bs_analytic = black_scholes_vanilla_dual_gamma(S, K_outputs, T, r, 0, imp_vol_atm)
implied_vol_guess = loc_vol_para.interpolate(x_values)

### TEST
K_test = K_outputs[119]
price_test = premium_outputs[119]
vol_test = np.linspace(0.2475, 0.2525, num=100, endpoint=True)
price_diff_test = black_scholes_vanilla(S, K_test, T, r, 0, vol_test) / price_test -1
print('K=',K_test, ' price=',price_test)
plt.plot(vol_test, price_diff_test)
plt.hlines(0, 0.2475, 0.2525)
plt.title('price vs vol')
plt.show()
### TEST END

implied_vol = black_scholes_vanilla_solve_vol(S, K_outputs[1:-1], T, r, 0, implied_vol_guess[1:-1], premium_outputs[1:-1])
premium_bs_implied_vol = black_scholes_vanilla(S, K_outputs[1:-1], T, r, 0, implied_vol)


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

