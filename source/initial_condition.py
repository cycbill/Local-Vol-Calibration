import numpy as np 

class InitialConditionFirstTenor():
    def __init__(_S):
        self.S = _S
    def compute(k):
        return np.maximum(1 - np.exp(k), 0)

class InitialConditionOtherTenors():
    def __init__(self, _k_inputs, _premium_inputs):
        self.k_inputs = _k_inputs
        self.premium_inputs = _premium_inputs
        self._interpolation_class = CubicSplineParameter1D(_k_inputs, _premium_inputs)
    def compute(self, k):
        result = self._interpolation_class.interpolate(k)
        return result