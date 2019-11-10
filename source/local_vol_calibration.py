#############################################################
# Calibrate local vol from calibration baskets on 1 pillar
# Input: 5 point quotes (k, lv)
# Process:
# (1) Fwd PDE with local vol, return k and lv series on last t grid
# (2) Interpolate on last t grid to get loc vol prices on 5 quoted k
# (3) Fwd PDE with constant implied vol, return k and lv series on last t grid
# (4) Interpolate on last t grid to get prices on 1 quoted k
# (5) Do (3), (4) for 5 times to get 5 bs prices
# (6) Set up cost function, initial loc vol guess, solve with BFGS.
# Output: 5 point quotes (k, lv)
#############################################################

import numpy as np 
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import norm

from tenor_market_data import TenorMarketData
from parameters import *
from initial_condition import *
from fwd_fdm import FDMCrankNicolsonNeumann
from black_scholes_formulas import black_scholes_vanilla


class LocalVolCalibration():
    def __init__(self, _x_min, _x_max, _J, _t_min, _t_max, _N, _tenor_mkt_data, _imp_vol_para, _init_cond_lv, _init_cond_bs):
        self.x_min = _x_min
        self.x_max = _x_max
        self.J = _J
        self.t_min = _t_min
        self.t_max = _t_max
        self.N = _N
        self.tenor_mkt_data = _tenor_mkt_data
        self.imp_vol_para = _imp_vol_para
        self.init_cond_lv = _init_cond_lv
        self.init_cond_bs = _init_cond_bs

        self.nb_quotes = len(self.imp_vol_para.x_inputs)
        self.prices_5quotes_bs = self.blk_sch_fwd_pde()
        self.solve_iter_count = 0


    def loc_vol_fwd_pde(self):
        fdm_lv = FDMCrankNicolsonNeumann(self.x_min, self.x_max, self.J, self.t_min, self.t_max, self.N, \
                                        self.tenor_mkt_data, self.loc_vol_para, self.init_cond_lv)
        price_grid, k_grid = fdm_lv.step_march()
        price_interpolator = PiecewiseLinearParameter1D(k_grid, price_grid)
        prices_5quotes_lv = price_interpolator.interpolate(self.loc_vol_para.x_inputs)
        return prices_5quotes_lv, price_grid, k_grid


    def blk_sch_fwd_pde(self):
        prices_5quotes_bs = np.zeros_like(self.imp_vol_para.x_inputs)

        for i, k_quote, imp_vol_quote in zip(range(self.nb_quotes), self.imp_vol_para.x_inputs, self.imp_vol_para.value_inputs):
            loc_vol_constant = ConstantParameter1D(imp_vol_quote)
            fdm_bs = FDMCrankNicolsonNeumann(self.x_min, self.x_max, self.J, self.t_min, self.t_max, self.N, \
                                            self.tenor_mkt_data, loc_vol_constant, self.init_cond_bs)
            price_grid, k_grid = fdm_bs.step_march()
            price_interpolator = PiecewiseLinearParameter1D(k_grid, price_grid)
            prices_5quotes_bs[i] = price_interpolator.interpolate(k_quote)
        
        return prices_5quotes_bs


    def panel_func(self):
        slopes = np.diff(self.loc_vol_para.value_inputs) / np.diff(self.loc_vol_para.x_inputs)
        convave_proxy = np.diff(slopes) * (-1.0)
        atm_vol = self.imp_vol_para.value_inputs[2]
        atm_K = self.tenor_mkt_data.fwd
        coeff_panel = 0.5 * (0.05 * atm_vol)**2 * self.tenor_mkt_data.T   # lambda in MX LV
        atm_price = self.tenor_mkt_data.black_scholes_price(atm_K, atm_vol) / self.tenor_mkt_data.spot   # used to compute alpha in MX LV, refer this formula to c(T,k) later
        alpha = (atm_price * np.sqrt(2 * np.pi)) / ((self.nb_quotes - 2) * np.sqrt(self.tenor_mkt_data.T))
        gamma_func = convave_proxy * norm.cdf(convave_proxy / alpha) + alpha * norm.pdf(convave_proxy / alpha)
        result = coeff_panel * np.sum(gamma_func**2)
        return result


    def cost_func(self, loc_vol_data):
        # we can extend the loc vol data to 7 points later
        self.loc_vol_para = PiecewiseLinearParameter1D(self.imp_vol_para.x_inputs, loc_vol_data)
        self.prices_5quotes_lv, self.price_grid_lv, self.k_grid = self.loc_vol_fwd_pde()
        cost = np.sum( (self.prices_5quotes_lv - self.prices_5quotes_bs)**2 ) + self.panel_func()
        self.solve_iter_count += 1
        print('Iter ', self.solve_iter_count, ' : ', loc_vol_data, cost)
        return cost


    def calibration(self, loc_vol_guess):
        x_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        solve_result = minimize(self.cost_func, loc_vol_guess, method='L-BFGS-B', bounds=x_bounds, tol=1.2e-08 )
        self.loc_vol_solved = solve_result.x
        return self.loc_vol_solved, self.prices_5quotes_lv, self.prices_5quotes_bs, self.price_grid_lv, self.k_grid


if __name__ == '__main__':
    S = 0.6
    r = 0.25
    T = 1

    ## K & Vatm inputs
    K_inputs = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
    #imp_vol_inputs = np.array([0.25, 0.22, 0.20, 0.19, 0.21])
    imp_vol_inputs = np.array([0.2, 0.21, 0.22, 0.21, 0.2])
    imp_vol_atm = imp_vol_inputs[2]
    loc_vol_guess = imp_vol_inputs
    #loc_vol_guess = np.array([0.25, 0.22, 0.20, 0.19, 0.21])

    F = S * np.exp(r*T)
    k_inputs = np.log(K_inputs / F)
    x_min = min( k_inputs[0] * 0.9, -5*imp_vol_atm*np.sqrt(T) )
    x_max = max( k_inputs[-1] * 1.1, 5*imp_vol_atm*np.sqrt(T) )
    J = 200
    t_min = 0
    t_max = T
    N = 200

    tenor_mkt_data = TenorMarketData(S, r, T)
    imp_vol_para = CubicSplineParameter1D(K_inputs, imp_vol_inputs)
    init_cond_lv = InitialConditionFirstTenor()     # testing purpose
    init_cond_bs = InitialConditionFirstTenor()     # testing purpose

    lv_calibrator = LocalVolCalibration(x_min, x_max, J, t_min, t_max, N, tenor_mkt_data, imp_vol_para, init_cond_lv, init_cond_bs)
    loc_vol_solved, prices_5quotes_lv, prices_5quotes_bs, price_grid_lv, k_grid = lv_calibrator.calibration(loc_vol_guess)
    print(loc_vol_solved)

