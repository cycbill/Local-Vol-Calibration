###################################################
# To compute the k_min and k_max on new pillar via extrapolation
###################################################

import numpy as np 
from scipy.stats import norm

from tenor_market_data import TenorMarketData
from implied_vol_class import ImpliedVolatility
from black_scholes_formulas import *

class NewPillarStrikeExtrapolation():
    def __init__(self, _tenor_mkt_data, _imp_vol_para):
        self.tenor_mkt_data = _tenor_mkt_data
        self.imp_vol_para = _imp_vol_para
        self.strike = self.imp_vol_para.K_inputs
        self.moneyness = self.strike / self.tenor_mkt_data.fwd
        self.norm_call_price = black_scholes_vanilla(self.tenor_mkt_data.spot, self.strike, self.tenor_mkt_data.T, 
                                                     self.tenor_mkt_data.rd, 0, self.imp_vol_para.value_inputs)