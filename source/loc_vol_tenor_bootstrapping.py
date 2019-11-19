#################################################
# Bootstrapping to solve local vol by tenor
# Input: r_quotes, r_tenors, rf_quotes, rf_tenors, imp_vol_quotes, imp_vol_tenors, imp_vol_strikes
# Output: loc_vol_surface
#################################################

import numpy as np 
from source.rate_curve_class import RateCurve
from implied_vol_class import ImpliedVolatility
from source.local_vol_init_guess_class import LocalVolatilityInitGuess
from source.new_pillar_strike_extrapolation import NewPillarStrikeExtrapolation

class VolDeltaScale():
    def __init__(self, _delta_value, _spotfwd_type, _premium_include):
        self.delta_value = _delta_value
        self.spotfwd_type = _spotfwd_type
        self.premium_include = _premium_include

def loc_vol_tenor_bootstrapping(spot, r_quotes, r_tenors, rf_quotes, rf_tenors, imp_vol_quotes, imp_vol_tenors, imp_vol_delta):
    r_para = RateCurve(r_tenors, r_quotes)
    rf_para = RateCurve(rf_tenors, rf_quotes)

    nb_tenors = len(imp_vol_tenors)
    nb_strikes = imp_vol_strikes.shape[1]
    mid = nb_strikes / 2
    imp_vol_atm = imp_vol_surface[:, mid]
    #imp_vol_surface = ImpliedVolatilitySurface(imp_vol_tenors, imp_vol_strikes, imp_vol_quotes)
    
    for i in range(nb_tenors):
        # build 'tenor market data' class
        tenor_mkt_data = TenorMarketData(spot, r_para, rf_para, imp_vol_tenors[i])

        # solve K and k from delta
        K_guess = np.repeat(F, 5)
        K_inputs = black_scholes_vanilla_solve_strike(callput, S, K_guess, T, r, 0, imp_vol_inputs, delta, 'fwd')

        # build 'implied vol' class
        ###### solve strike from delta
        imp_vol_para = 


        loc_vol_init_guess_class = LocalVolatilityInitGuess(imp_vol_strikes[i,:], imp_vol_quotes[i,:], imp_vol_atm[i], imp_vol_tenors[i])
        loc_vol_guess = loc_vol_init_guess_class.compute_init_guess()

        NewPillarStrikeExtrapolation()
        k_min_extrplt, k_max_extrplt = 



if __name__ == '__main__':
    spot = 0.6
    trade_maturity = 1.0

    # Rate curve data
    r_tenors = np.array([0.01, 10.0])
    r_quotes = np.array([0.05, 0.05])
    rf_tenors = np.array([0.01, 10.0])
    rf_quotes = np.array([0.0, 0.0])

    # Vol surface data
    imp_vol_tenors = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0])
    nb_tenors = len(imp_vol_tenors)

    delta_value = np.array([0.90, 0.75, 0.50, 0.25, 0.10])
    spotfwd_type = ['spot', 'spot', 'spot', 'spot', 'spot', 'spot', 'spot', 'fwd', 'fwd', 'fwd']
    premium_include = ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no']
    delta_scale = VolDeltaScale

    imp_vol_1pillar = np.array([0.25, 0.23, 0.21, 0.19, 0.22])
    imp_vol_quotes = np.repeat(imp_vol_1pillar, (nb_tenors, 1))
