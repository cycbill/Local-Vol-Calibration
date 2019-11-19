###################################################
# To compute the k_min and k_max on new pillar via extrapolation
###################################################

import numpy as np 
from scipy.stats import norm
from scipy.optimize import newton

from tenor_market_data import TenorMarketData
from implied_vol_class import ImpliedVolatility
from black_scholes_formulas import *

class NewPillarStrikeExtrapolation():
    def __init__(self, _tenor_mkt_data, _imp_vol_para):
        self.tenor_mkt_data = _tenor_mkt_data
        self.imp_vol_para = _imp_vol_para
        self.strike = self.imp_vol_para.K_inputs
        self.nb_quotes = len(self.strike)
        self.compute_initilization()

        

    def compute_initilization(self):
        self.norm_call_price = np.zeros(self.nb_quotes + 1)
        self.norm_call_price[0] = 1
        self.norm_call_price[1:] = self.compute_norm_call()

        self.moneyness = np.zeros(self.nb_quotes + 1)
        self.moneyness[1:] = self.strike / self.tenor_mkt_data.fwd

        self.slope = np.zeros(self.nb_quotes + 1)
        self.slope[0] = -1
        self.slope[1:] = np.diff(self.norm_call_price) / np.diff(self.moneyness)

        
    def compute_norm_call(self):
        call_price = black_scholes_vanilla(1, self.tenor_mkt_data.spot, self.strike, self.tenor_mkt_data.T, 
                    self.tenor_mkt_data.r, self.tenor_mkt_data.rf, self.imp_vol_para.value_inputs) 
        norm_call_price = call_price / (self.tenor_mkt_data.fwd * self.tenor_mkt_data.DF_r)
        return norm_call_price

    def put_quantile_slope_func(self, sigma):
        digital_price = - (self.slope[1] + self.slope[2]) / 2
        self.d_minus_1 = norm.ppf(digital_price)
        eta_low = - (self.d_minus_1 + sigma) 
        func_result = (eta_low)**2 / 2 + np.log(norm.cdf(eta_low)) - np.log((self.slope[2] - self.slope[1]) / 2) - (d_minus)**2 / 2
        return func_result

    def compute_otm_put_sigma(self):
        mid = self.nb_quotes / 2
        sigma_guess = self.imp_vol_para.value_inputs[mid] * np.sqrt(self.tenor_mkt_data.T)
        sigma_solved = newton(self.put_quantile_slope_func, sigma_guess)

        f = self.moneyness[1] * np.exp(sigma_solved * ( self.d_minus_1 + sigma_solved / 2))

        d_minus_0 = np.log(f / self.moneyness[0]) / sigma_solved - sigma_solved / 2
        