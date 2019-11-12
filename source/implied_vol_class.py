#####################################
# Build the implied vol class
#####################################

import numpy as np 

from parameters import CubicSplineParameter1D

class ImpliedVolatility():
    def __init__(self, _K_inputs, _k_inputs, _imp_vol_inputs):
        self.K_inputs = _K_inputs
        self.x_inputs = _k_inputs
        self.value_inputs = _imp_vol_inputs
        self.imp_vol_para = CubicSplineParameter1D(self.x_inputs, self.value_inputs)

    def interpolate(self, x):
        return self.imp_vol_para.interpolate(x)