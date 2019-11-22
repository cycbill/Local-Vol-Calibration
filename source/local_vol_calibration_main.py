import numpy as np 
import xlwings as xw 

from rate_curve_class import RateCurve
from strike_grid_discretization import StrikeGridsAllTenors
from tenor_market_data import TenorMarketData
from implied_vol_class import ImpliedVolatility
from compute_sum_sqr_vol_T import compute_sum_sqr_vol_T
from new_pillar_strike_extrapolation import NewPillarStrikeExtrapolation
from compute_maturity_grid import compute_maturity_grid

np.set_printoptions(linewidth=150)
wb = xw.Book('LocVol Parameters.xlsx')
sht = wb.sheets['IRSMFORM']

## Read data from file
S = sht.range('B2').value

r_tenors = sht.range('B6').options(np.array, expand='down').value
r_quotes = sht.range('C6').options(np.array, expand='down').value
rf_tenors = sht.range('E6').options(np.array, expand='down').value
rf_quotes = sht.range('F6').options(np.array, expand='down').value
csc_tenors = sht.range('H6').options(np.array, expand='down').value
csc_quotes = sht.range('I6').options(np.array, expand='down').value

imp_vol_tenors = sht.range('K6').options(np.array, expand='down').value
imp_vol_strikes = sht.range('N6').options(np.array, expand='table').value
imp_vol_quotes = sht.range('T6').options(np.array, expand='table').value


## Whole Set Data initialization
r_para = RateCurve(r_tenors, r_quotes)
rf_para = RateCurve(rf_tenors, rf_quotes)
csc_para = RateCurve(csc_tenors, csc_quotes)

nb_tenors = len(imp_vol_tenors)
nb_strikes = imp_vol_strikes.shape[1]
mid = nb_strikes / 2

strike_grid_info = StrikeGridsAllTenors(nb_tenors)


## Pillar Data initialization
i = 0     # which tenor we are looking at
T = imp_vol_tenors[i]
tenor_mkt_data = TenorMarketData(S, r_para, rf_para, csc_para, T)

K_inputs = imp_vol_strikes[i, :]
k_inputs = np.log(K_inputs / tenor_mkt_data.fwd)
imp_vol_inputs = imp_vol_quotes[i, :]
imp_vol_para = ImpliedVolatility(K_inputs, k_inputs, imp_vol_inputs)

sum_sqr_vol_T = compute_sum_sqr_vol_T(imp_vol_quotes, imp_vol_tenors)


## Compute k_min, k_max, dk, Nk
new_pillar_extrplt = NewPillarStrikeExtrapolation(tenor_mkt_data, imp_vol_para)
k_min_extrplt, k_max_extrplt = new_pillar_extrplt.compute_extreme_strikes()

k_min, k_max, dk, Nk = strike_grid_info.strike_grid_discretization(i, k_min_extrplt, k_max_extrplt, 
                                                                    imp_vol_para, tenor_mkt_data, sum_sqr_vol_T[i])


## Compute t_min, t_max, dT, NT
t_min, t_max, dt, NT = compute_maturity_grid(i, imp_vol_tenors)

print('maturity: ', T)
t_grids = np.linspace(t_min, t_max, NT, endpoint=True)
print('t_grids: ', t_grids)
print('k_min: {}, k_max: {}, dk: {}, Nk: {}.'.format(k_min, k_max, dk, Nk))
print('NT: ', NT)

print('k_min_extrplt: {}, k_max_extrplt: {}.'.format(k_min_extrplt, k_max_extrplt))

## Local vol initial guess