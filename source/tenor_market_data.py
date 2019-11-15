import numpy as np
from black_scholes_formulas import black_scholes_vanilla

class TenorMarketData():
    def __init__(self, _spot, _r, _rf, _T):
        self.spot = _spot
        self.r = _r
        self.rf = _rf
        self.T = _T
        self.fwd = self.spot * np.exp(self.r * self.T)

    def black_scholes_price(self, callput, K, vol):
        r_T = self.r.interpolate(T)
        rf_T = self.rf.interpolate(T)
        return black_scholes_vanilla(callput, self.spot, K, self.T, r_T, rf_T, vol)
