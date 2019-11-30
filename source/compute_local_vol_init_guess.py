#####################################
# Compute initial guess of local vol
#####################################

import numpy as np 
import matplotlib.pyplot as plt

from parameters import PiecewiseLinearParameter1D


def compute_local_vol_init_guess(k_quotes, imp_vol_quotes, T):
    m = len(k_quotes)
    first_deriv = np.zeros(m)
    second_deriv = np.zeros(m)

    diff_strike = np.diff(k_quotes)
    diff_imp_vol = np.diff(imp_vol_quotes)

    sum_adjucant_strike = diff_strike[:-1] + diff_strike[1:]

    first_deriv[0] = diff_imp_vol[0] / diff_strike[0]
    first_deriv[-1] = diff_imp_vol[-1] / diff_strike[-1]
    first_deriv[1:-1] = ( diff_imp_vol[:-1] * diff_strike[1:] / diff_strike[:-1] + diff_imp_vol[1:] * diff_strike[:-1] / diff_strike[1:] ) / sum_adjucant_strike

    second_deriv[0] = 0
    second_deriv[-1] = 0
    second_deriv[1:-1] = 2.0 * (imp_vol_quotes[:-2] / sum_adjucant_strike / diff_strike[:-1] \
                                - imp_vol_quotes[1:-1] / diff_strike[1:] / diff_strike[:-1] \
                                + imp_vol_quotes[2:] / sum_adjucant_strike / diff_strike[1:] )

    strike_div_vol = k_quotes / imp_vol_quotes
    vol_times_T = imp_vol_quotes * T

    loc_vol_square = imp_vol_quotes**2 / ( (1 - 2 * strike_div_vol * first_deriv) \
                                                + (strike_div_vol**2 - vol_times_T**2 / 4) * first_deriv**2 
                                                + vol_times_T * second_deriv )
    
    result = np.sqrt(loc_vol_square)
    return result