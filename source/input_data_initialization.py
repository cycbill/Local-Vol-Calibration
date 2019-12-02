import numpy as np 
import xlwings as xw 

from rate_curve_class import RateCurve
from strike_grid_discretization import StrikeGridsAllTenors
from tenor_market_data import TenorMarketData
from implied_vol_class import ImpliedVolatility
from compute_sum_sqr_vol_T import compute_sum_sqr_vol_T
from new_pillar_strike_extrapolation import NewPillarStrikeExtrapolation
from compute_maturity_grid import compute_maturity_grid
from compute_local_vol_init_guess import compute_local_vol_init_guess

def input_data_initialization():
    np.set_printoptions(linewidth=150)
    wb = xw.Book('LocVol Parameters.xlsx')
    #wb = xw.Book(r'source\LocVol Parameters.xlsx')
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


    ## Build up rate curve class
    r_para = RateCurve(r_tenors, r_quotes)
    rf_para = RateCurve(rf_tenors, rf_quotes)
    csc_para = RateCurve(csc_tenors, csc_quotes)

    return S, r_para, rf_para, csc_para, imp_vol_tenors, imp_vol_strikes, imp_vol_quotes
