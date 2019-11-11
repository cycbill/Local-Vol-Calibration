#####################################
# Compute initial guess of local vol
#####################################

import numpy as np 
import matplotlib.pyplot as plt

from parameters import PiecewiseLinearParameter1D

class LocalVolatilityInitGuess():
    def __init__(self, _k_quotes, _loc_vol_quotes, _imp_vol_atm, _T):
        self.x_inputs = _k_quotes
        self.value_inputs = _loc_vol_quotes
        self.imp_vol_atm = _imp_vol_atm
        self.T = _T
        self.m = len(self.x_inputs)
    

    def compute_init_guess(self):
        first_deriv = np.zeros(self.m)
        second_deriv = np.zeros(self.m)

        diff_strike = np.diff(self.x_inputs)
        diff_imp_vol = np.diff(self.value_inputs)

        sum_adjucant_strike = diff_strike[:-1] + diff_strike[1:]

        first_deriv[0] = diff_imp_vol[0] / diff_strike[0]
        first_deriv[-1] = diff_imp_vol[-1] / diff_strike[-1]
        first_deriv[1:-1] = ( diff_imp_vol[:-1] * diff_strike[1:] / diff_strike[:-1] + diff_imp_vol[1:] * diff_strike[:-1] / diff_strike[1:] ) / sum_adjucant_strike

        second_deriv[0] = 0
        second_deriv[-1] = 0
        second_deriv[1:-1] = 2.0 * (self.value_inputs[:-2] / sum_adjucant_strike / diff_strike[:-1] \
                                    - self.value_inputs[1:-1] / diff_strike[1:] / diff_strike[:-1] \
                                    + self.value_inputs[2:] / sum_adjucant_strike / diff_strike[1:] )

        strike_div_vol = self.x_inputs / self.value_inputs
        vol_times_T = self.value_inputs**2 * self.T**2

        loc_vol_square = self.value_inputs**2 / ( (1 - 2 * strike_div_vol * first_deriv) \
                                                   + (strike_div_vol**2 - vol_times_T**2 / 4) * first_deriv**2 
                                                   + vol_times_T * second_deriv )
        
        result = np.sqrt(loc_vol_square)
        return result