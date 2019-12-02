import numpy as np 

from parameters import CubicSplineLinearExtrpParameter1D
from black_scholes_formulas import black_scholes_vanilla_fwd_norm

class InitialConditionFirstTenor():
    def compute(self, k):
        return np.maximum(1 - np.exp(k), 0)

class InitialConditionOtherTenors():
    def __init__(self, _k_inputs, _premium_inputs):
        self.k_inputs = _k_inputs
        self.premium_inputs = _premium_inputs
        self._interpolation_class = CubicSplineLinearExtrpParameter1D(_k_inputs, _premium_inputs)
    def compute(self, k):
        result = self._interpolation_class.interpolate(k)
        return result


class InitialConditionBlackScholes():
    def __init__(self, i, _T_prev, _tenor_mkt_data, _imp_vol_para):
        self.callput = 1
        self.T = _T_prev
        self.r = _tenor_mkt_data.r
        self.vol = _imp_vol_para.value_inputs[i]
    def compute(self, k):
        #K = np.exp(k) * self.fwd
        #norm_call_price = black_scholes_vanilla_fwd_norm(self.callput, self.fwd, K, self.T, self.r, self.rf, self.vol)
        norm_call_price = black_scholes_vanilla_fwd_norm(self.callput, k, self.T, self.r, self.vol)
        return norm_call_price
