import numpy as np

class CalibrationBasketVanillaOption():
    def __init__(self, _spot, _r, _T, _loc_vol):
        self.spot = _spot
        self.r = _r
        self.T = _T
        self.loc_vol = _loc_vol
        self.fwd = self.spot * np.exp(self.r * self.T)
        
    def payoff_by_logmoney(self, logmoney):
        return np.maximum(self.spot - np.exp(logmoney), 0)