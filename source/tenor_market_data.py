import numpy as np
from black_scholes_formulas import black_scholes_vanilla

class TenorMarketData():
    def __init__(self, _spot, _r, _T):
        self.spot = _spot
        self.r = _r
        self.T = _T
        self.fwd = self.spot * np.exp(self.r * self.T)

    def black_scholes_price(self, callput, K, vol):
        return black_scholes_vanilla(callput, self.spot, K, self.T, self.r, 0, vol)
