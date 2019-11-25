import numpy as np
from black_scholes_formulas import black_scholes_vanilla

class TenorMarketData():
    def __init__(self, _spot, _r_para, _rf_para, _csc_para, _T):
        self.spot = _spot
        self.r_para = _r_para
        self.rf_para = _rf_para
        self.csc_para = _csc_para
        self.T = _T
        self.r = _r_para.interpolate(self.T)
        self.rf = _rf_para.interpolate(self.T)
        self.csc = _csc_para.interpolate(self.T)
        self.CF_r = _r_para.CF(0, self.T)
        self.CF_rf = _rf_para.CF(0, self.T)
        self.CF_csc = _csc_para.CF(0, self.T)
        self.DF_r = 1 / self.CF_r
        self.DF_rf = 1 / self.CF_rf
        self.DF_csc = 1 / self.CF_csc
        self.fwd = self.spot * self.CF_r / self.CF_rf / self.CF_csc


    def black_scholes_price(self, callput, K, vol):

        return black_scholes_vanilla(callput, self.spot, K, self.T, self.r, self.rf, vol)
