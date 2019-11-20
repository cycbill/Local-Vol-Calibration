import numpy as np 
import xlwings as xw 

from rate_curve_class import RateCurve
from tenor_market_data import TenorMarketData

wb = xw.Book('LocVol Parameters.xlsx')
sht = wb.sheets['IRSMFORM']

## Read data from file
S = sht.range('B2').value

r_tenors = sht.range('B6').options(np.array, expand='down').value
r_quotes = sht.range('C6').options(np.array, expand='down').value
rf_tenors = sht.range('E6').options(np.array, expand='down').value
rf_quotes = sht.range('F6').options(np.array, expand='down').value

imp_vol_tenors = sht.range('K6').options(np.array, expand='down').value
imp_vol_quotes = sht.range('T6').options(np.array, expand='right').value

atm_definition = 'dns'
premium_include = 'no'
delta_spotfwd = 'spot'
delta_quotes = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
delta_callput = ['call', 'call', 'put', 'put', 'put']

## Data initialization
tenor_i = 0     # which tenor we are looking at

r_para = RateCurve(r_tenors, r_quotes)
rf_para = RateCurve(rf_tenors, rf_quotes)

T = imp_vol_tenors[tenor_i]

tenor_mkt_data = TenorMarketData(S, r_para, rf_para, T)

nb_tenors = len(imp_vol_tenors)
nb_strikes = len(imp_vol_quotes)

mid = nb_strikes / 2
atm_imp_vol = imp_vol_quotes[mid]
