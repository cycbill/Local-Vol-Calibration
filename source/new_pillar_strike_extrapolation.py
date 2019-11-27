###################################################
# To compute the k_min and k_max on new pillar via extrapolation
###################################################

import numpy as np 
from scipy.stats import norm
from scipy.optimize import newton, brenth
import matplotlib.pyplot as plt

from tenor_market_data import TenorMarketData
from implied_vol_class import ImpliedVolatility
from black_scholes_formulas import *

class NewPillarStrikeExtrapolation():
    def __init__(self, _tenor_mkt_data, _imp_vol_para):
        self.ALPHA = 2.1e-7
        self.tenor_mkt_data = _tenor_mkt_data
        self.imp_vol_para = _imp_vol_para
        self.strike = self.imp_vol_para.K_inputs
        self.M = len(self.strike)
        self.compute_initilization()

    def compute_initilization(self):
        self.norm_call_price = np.zeros(self.M + 1)
        self.norm_call_price[0] = 1
        self.norm_call_price[1:] = self.compute_norm_call()

        self.moneyness = np.zeros(self.M + 1)
        self.moneyness[1:] = self.strike / self.tenor_mkt_data.fwd

        self.slope = np.zeros(self.M + 1)
        self.slope[0] = -1
        self.slope[1:] = np.diff(self.norm_call_price) / np.diff(self.moneyness)

        mid = int((self.M - 1) / 2)
        self.sigma_guess = self.imp_vol_para.value_inputs[mid] * np.sqrt(self.tenor_mkt_data.T)

        
    def compute_norm_call(self):
        call_price = black_scholes_vanilla_fwd(1, self.tenor_mkt_data.fwd, self.strike, self.tenor_mkt_data.T, 
                    self.tenor_mkt_data.r, self.tenor_mkt_data.rf, self.imp_vol_para.value_inputs) 
        norm_call_price = call_price / (self.tenor_mkt_data.fwd * self.tenor_mkt_data.DF_r)
        return norm_call_price

    def digital_extrapolation(self, moneyness, f, sigma):
        d_minus = np.log(f / moneyness) / sigma - 0.5 * sigma
        digital_price = norm.cdf(d_minus)
        return digital_price

    def put_quantile_slope_func(self, sigma):
        digital_price = - (self.slope[1] + self.slope[2]) / 2
        self.d_minus_1 = norm.ppf(digital_price)
        eta_low = - (self.d_minus_1 + sigma) 
        func_result = (eta_low)**2 / 2 + np.log(norm.cdf(eta_low)) \
                        - np.log((self.slope[2] - self.slope[1]) / 2) - (self.d_minus_1)**2 / 2
        return func_result

    def otm_put_extrapolation(self):
        sigma_solved = newton(self.put_quantile_slope_func, self.sigma_guess)

        f = self.moneyness[1] * np.exp(sigma_solved * ( self.d_minus_1 + sigma_solved / 2))
        return f, sigma_solved

    def call_quantile_slope_func(self, sigma):
        digital_price = - self.slope[-1] / 2
        self.d_minus_M = norm.ppf(digital_price)
        eta_high = self.d_minus_M + sigma
        func_result = (eta_high)**2 / 2 + np.log(norm.cdf(eta_high)) \
                        - np.log(self.norm_call_price[-1] / self.moneyness[-1] - self.slope[-1] / 2) \
                        + self.d_minus_M**2
        return func_result

    def otm_call_price_diff(self, sigma):
        digital_price = - self.slope[-1] / 2
        self.d_minus_M = norm.ppf(digital_price)
        self.d_plus_M = self.d_minus_M + sigma

        f = self.moneyness[-1] * np.exp(sigma * (self.d_minus_M + sigma / 2))
        call_func = f * norm.cdf(self.d_plus_M) - self.moneyness[-1] * norm.cdf(self.d_minus_M)
        call_price_diff = call_func - self.norm_call_price[-1]
        return call_price_diff


    def otm_call_extrapolation(self):
        ############ TEST
        '''
        xmin = 0.001
        xmax = 0.025
        print(self.otm_call_price_diff(xmin), self.otm_call_price_diff(xmax))
        sigma_test = np.linspace(xmin, xmax, 50)
        func_test = self.otm_call_price_diff(sigma_test)
        plt.plot(sigma_test, func_test, '.-')
        plt.hlines(0,xmin,xmax)
        plt.show()
        '''
        ############ TEST END
        sigma_solved = newton(self.otm_call_price_diff, self.sigma_guess)
        print(sigma_solved)
        #sigma_solved = newton(self.call_quantile_slope_func, self.sigma_guess)

        f = self.moneyness[-1] * np.exp(sigma_solved * (self.d_minus_M + sigma_solved / 2))
        return f, sigma_solved


    def compute_extreme_strikes(self):
        f_otmput, sigma_otmput = self.otm_put_extrapolation()
        dig_extrplt_func1 = lambda moneyness: self.digital_extrapolation(moneyness, f_otmput, sigma_otmput) - (1 - self.ALPHA)
        moneyness_min = newton(dig_extrplt_func1, self.moneyness[1])
        k_min = np.log(moneyness_min)
        k_min2 = np.log(f_otmput / (np.exp(sigma_otmput*(0.5*sigma_otmput+5))))

        f_otmcall, sigma_otmcall = self.otm_call_extrapolation()
        dig_extrplt_func2 = lambda moneyness: self.digital_extrapolation(moneyness, f_otmcall, sigma_otmcall) - self.ALPHA
        moneyness_max = newton(dig_extrplt_func2, self.moneyness[-1])
        k_max = np.log(moneyness_max)
        k_max2 = np.log(f_otmcall / (np.exp(sigma_otmcall*(0.5*sigma_otmcall-5))))

        return k_min, k_max
    