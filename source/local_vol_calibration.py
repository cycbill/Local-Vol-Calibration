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
import matplotlib.pyplot as plt

from input_data_initialization import input_data_initialization
from rate_curve_class import RateCurve
from tenor_market_data import TenorMarketData
from parameters import *
from implied_vol_class import ImpliedVolatility
from local_vol_class import LocalVolatility
from initial_condition import *
from fwd_fdm import FDMCrankNicolsonNeumann
from black_scholes_formulas import *
from read_write_excel import read_from_excel


class LocalVolCalibration():
    def __init__(self, _x_min, _x_max, _x_values, _J, _t_min, _t_max, _t_values, _theta, _N, _tenor_mkt_data, _imp_vol_para, _init_cond_lv, _prices_5quotes_bs, _debug):
        self.x_min = _x_min
        self.x_max = _x_max
        self.x_values = _x_values
        self.J = _J
        self.t_min = _t_min
        self.t_max = _t_max
        self.t_values = _t_values
        self.theta = _theta
        self.N = _N
        self.tenor_mkt_data = _tenor_mkt_data
        self.imp_vol_para = _imp_vol_para
        self.init_cond_lv = _init_cond_lv
        self.nb_quotes = len(self.imp_vol_para.x_inputs)
        self.prices_5quotes_bs = _prices_5quotes_bs
        self.solve_iter_count = 0
        self.match_cost = 0
        self.panel_cost = 0
        self.total_cost = 0
        self.debug = _debug

    '''
    def blk_sch_fwd_pde(self):
        prices_5quotes_bs = np.zeros_like(self.imp_vol_para.x_inputs)

        for i, k_quote, imp_vol_quote in zip(range(self.nb_quotes), self.imp_vol_para.x_inputs, self.imp_vol_para.value_inputs):
            ## Use constant vol
            loc_vol_constant = ConstantParameter1D(imp_vol_quote)
            ## Define initial condition
            if self.pillar_nb == 0:
                self.init_cond_bs = InitialConditionFirstTenor()
            else:
                self.init_cond_bs = InitialConditionBlackScholes(i, self.tenor_mkt_data, self.imp_vol_para)
            ## Build pde class
            fdm_bs = FDMCrankNicolsonNeumann(self.x_min, self.x_max, self.x_values, self.J, self.t_min, self.t_max, self.t_values, self.theta, self.N, \
                                            self.tenor_mkt_data, loc_vol_constant, self.init_cond_bs)
            ## PDE diffusion
            price_matrix, k_grid = fdm_bs.step_march()
            price_grid = price_matrix[-1, :]
            price_interpolator = PiecewiseLinearParameter1D(k_grid, price_grid)
            prices_5quotes_bs[i] = price_interpolator.interpolate(k_quote)
        
        return prices_5quotes_bs
    '''

    def loc_vol_fwd_pde(self):
        fdm_lv = FDMCrankNicolsonNeumann(self.x_min, self.x_max, self.x_values, self.J, self.t_min, self.t_max, self.t_values, self.theta, self.N, \
                                        self.tenor_mkt_data, self.loc_vol_para, self.init_cond_lv)
        price_matrix = fdm_lv.step_march()
        price_grid = price_matrix[-1, :]
        price_interpolator = PiecewiseLinearParameter1D(self.x_values, price_grid)
        prices_5quotes_lv = price_interpolator.interpolate(self.loc_vol_para.x_inputs)
        return prices_5quotes_lv, price_grid


    def panel_func(self):
        slopes = np.diff(self.loc_vol_para.value_inputs) / np.diff(self.loc_vol_para.x_inputs)
        convave_proxy = np.diff(slopes) * (-1.0)
        atm_vol = self.imp_vol_para.value_inputs[2]
        atm_K = self.tenor_mkt_data.fwd
        coeff_panel = 0.5 * (0.05 * atm_vol)**2 * self.tenor_mkt_data.T   # lambda in MX LV
        atm_price = self.tenor_mkt_data.black_scholes_price(1, atm_K, atm_vol) / self.tenor_mkt_data.spot   # used to compute alpha in MX LV, refer this formula to c(T,k) later
        alpha = (atm_price * np.sqrt(2 * np.pi)) / ((self.nb_quotes - 2) * np.sqrt(self.tenor_mkt_data.T))
        gamma_func = convave_proxy * norm.cdf(convave_proxy / alpha) + alpha * norm.pdf(convave_proxy / alpha)
        result = coeff_panel * np.sum(gamma_func**2)
        return result


    def cost_func(self, loc_vol_data):
        self.loc_vol_para = LocalVolatility(self.imp_vol_para.x_inputs, loc_vol_data, self.imp_vol_para.imp_vol_atm, self.tenor_mkt_data.T)

        self.prices_5quotes_lv, self.price_grid_lv = self.loc_vol_fwd_pde()
        self.match_cost = np.sum( (self.prices_5quotes_lv - self.prices_5quotes_bs)**2  ) 
        self.panel_cost = self.panel_func()
        self.total_cost = self.match_cost + self.panel_cost

        self.solve_iter_count += 1

        if self.debug == True:
            print('Iter:       ', self.solve_iter_count)
            print('Loc vol:    ', self.loc_vol_para.value_extend)
            print('LV prices:  ', self.prices_5quotes_lv)
            print('BS prices:  ', self.prices_5quotes_bs)
            print('Match cost: ', self.match_cost, ' Panel cost: ', self.panel_cost, ' Total cost: ', self.total_cost)
            print('\n')
        return self.total_cost


    def calibration(self, loc_vol_guess):
        x_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        solve_result = minimize(self.cost_func, loc_vol_guess, method='L-BFGS-B', bounds=x_bounds, tol=1.2e-08 )
        self.loc_vol_solved = solve_result.x
        return self.loc_vol_solved, self.prices_5quotes_lv, self.price_grid_lv


if __name__ == '__main__':
    np.set_printoptions(linewidth=150)
    callput = 1
    S, r_para, rf_para, csc_para, imp_vol_tenors, imp_vol_strikes, imp_vol_quotes = input_data_initialization()
    T = 0.0191780821918
    pillar_nb = 1

    #T_inputs = np.array([0.01, 10.0])
    #r_inputs = np.array([0.05, 0.05])
    #rf_inputs = np.array([0.0, 0.0])
    #r_para = RateCurve(T_inputs, r_inputs)
    #rf_para = RateCurve(T_inputs, rf_inputs)
    #csc_para = RateCurve(T_inputs, rf_inputs)
    tenor_mkt_data = TenorMarketData(S, r_para, rf_para, csc_para, T)

    #delta = np.array([0.90, 0.75, 0.50, 0.25, 0.10])

    ## Imp Vol inputs
    #imp_vol_inputs = np.array([0.25, 0.22, 0.20, 0.19, 0.21])
    imp_vol_inputs = imp_vol_quotes[pillar_nb,:]
    imp_vol_atm = imp_vol_inputs[2]

    ## Loc vol inputs
    loc_vol_guess =  np.array([0.11760865, 0.11173581, 0.11154483, 0.11963189, 0.12788399])


    K_inputs = imp_vol_strikes[pillar_nb, :]
    k_inputs = np.log(K_inputs / tenor_mkt_data.fwd)


    J = 98
    x_min = -0.09391736906882849
    x_max = 0.10109773583359967
    x_values = np.linspace(x_min, x_max, J, endpoint=True)
    N = 10
    t_min = 0.0027397260274
    t_max = 0.0191780821918
    t_values = np.linspace(t_min, t_max, N, endpoint=True)
    theta = np.repeat(0.5, N)
    #theta[0] = 0
    #theta[1:5] = [1.0, 1.0, 1.0, 1.0]
    prices_5quotes_bs = np.array([0.02080577, 0.01279167, 0.00624299, 0.00240147, 0.00078891])

    imp_vol_para = ImpliedVolatility(K_inputs, k_inputs, imp_vol_inputs)

    k_prev, price_prev = read_from_excel()
    init_cond_lv = InitialConditionOtherTenors(k_prev, price_prev)     # testing purpose
    price_test = init_cond_lv.compute(x_values)
    plt.plot(x_values, price_test)
    plt.title('interpolate')
    plt.show()

    lv_calibrator = LocalVolCalibration(x_min, x_max, x_values, J, t_min, t_max, t_values, theta, N, tenor_mkt_data, imp_vol_para, init_cond_lv, prices_5quotes_bs, True)
    loc_vol_solved, prices_5quotes_lv, price_grid_lv = lv_calibrator.calibration(loc_vol_guess)
    
    
    prices_lv = prices_5quotes_lv * tenor_mkt_data.fwd * np.exp(-tenor_mkt_data.r*T)
    prices_bs = prices_5quotes_bs * tenor_mkt_data.fwd * np.exp(-tenor_mkt_data.r*T)
    premium_bs = black_scholes_vanilla(callput, S, K_inputs, T, tenor_mkt_data.r, tenor_mkt_data.rf, imp_vol_inputs)
    print('LV pde price: ', prices_lv)
    print('BS pde price: ', prices_bs)
    print('BS cls price: ', premium_bs)
    print('Local Vol:    ', loc_vol_solved)

    