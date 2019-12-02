#####################################
# Build the implied vol class
#####################################

import numpy as np 

from parameters import CubicSplineParameter1D

class ImpliedVolatility(CubicSplineParameter1D):
    def __init__(self, _K_inputs, _k_inputs, _imp_vol_inputs):
        self.K_inputs = _K_inputs
        self.x_inputs = _k_inputs
        self.value_inputs = _imp_vol_inputs

        mid = int((len(self.value_inputs) - 1) / 2)
        self.imp_vol_atm = self.value_inputs[mid]