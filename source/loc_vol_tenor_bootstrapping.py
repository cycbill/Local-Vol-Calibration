#################################################
# Bootstrapping to solve local vol by tenor
# Input: r_quotes, r_tenors, rf_quotes, rf_tenors, imp_vol_quotes, imp_vol_tenors, imp_vol_strikes
# Output: loc_vol_surface
#################################################

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from input_data_initialization import input_data_initialization
from strike_grid_discretization import StrikeGridsAllTenors
from tenor_market_data import TenorMarketData
from implied_vol_class import ImpliedVolatility
from compute_sum_sqr_vol_T import compute_sum_sqr_vol_T
from new_pillar_strike_extrapolation import NewPillarStrikeExtrapolation
from compute_maturity_grid import compute_maturity_grid
from compute_local_vol_init_guess import compute_local_vol_init_guess
from black_scholes_formulas import black_scholes_vanilla, black_scholes_vanilla_fwd_norm
from blk_sch_fwd_pde import blk_sch_fwd_pde
from local_vol_calibration import LocalVolCalibration
from initial_condition import InitialConditionFirstTenor, InitialConditionOtherTenors
from read_write_excel import print_to_excel


def loc_vol_tenor_bootstrapping():
    
    ## Whole Set Data initialization
    S, r_para, rf_para, csc_para, imp_vol_tenors, imp_vol_strikes, imp_vol_quotes = input_data_initialization()

    nb_tenors = len(imp_vol_tenors)
    nb_strikes = imp_vol_strikes.shape[1]
    mid = nb_strikes / 2
    
    strike_grid_info = StrikeGridsAllTenors(nb_tenors)

    loc_vol_all_tenors = np.zeros_like(imp_vol_quotes)  # to store solved-out local vols
    price_lv_all_tenors = np.zeros_like(imp_vol_quotes)  # to store LV fwd pde prices
    price_bs_all_tenors = np.zeros_like(imp_vol_quotes)  # to store BS fwd pde prices
    price_cls_all_tenors = np.zeros_like(imp_vol_quotes)  # to store BS analytical prices
    k_all_tenors = np.zeros_like(imp_vol_quotes)  # to store BS fwd pde prices
    
    print('################### Start Bootstrapping ###################\n')
    ## Bootstrapping on tenor to calibrate local vol
    for i in range(nb_tenors):
        print('Pillar: ', i)
        ## Pillar Data initialization
        T = imp_vol_tenors[i]
        if i == 0:
            T_prev = 0
        else:
            T_prev = imp_vol_tenors[i-1]

        tenor_mkt_data = TenorMarketData(S, r_para, rf_para, csc_para, T)

        K_inputs = imp_vol_strikes[i, :]
        k_inputs = np.log(K_inputs / tenor_mkt_data.fwd)
        imp_vol_inputs = imp_vol_quotes[i, :]

        imp_vol_para = ImpliedVolatility(K_inputs, k_inputs, imp_vol_inputs)

        sum_sqr_vol_T = compute_sum_sqr_vol_T(imp_vol_quotes, imp_vol_tenors)


        ## Compute k_min, k_max, dk, Nk
        new_pillar_extrplt = NewPillarStrikeExtrapolation(tenor_mkt_data, imp_vol_para)
        k_min_extrplt, k_max_extrplt = new_pillar_extrplt.compute_extreme_strikes()

        k_min, k_max, dk, Nk = strike_grid_info.strike_grid_discretization(i, k_min_extrplt, k_max_extrplt, 
                                                                            imp_vol_para, tenor_mkt_data, sum_sqr_vol_T[i])
        k_grids = np.linspace(k_min, k_max, Nk, endpoint=True)
        print('k_min: {}, k_max: {}, dk: {}, Nk: {}.'.format(k_min, k_max, dk, Nk))

        ## Compute t_min, t_max, dT, NT
        t_min, t_max, dt, NT = compute_maturity_grid(i, imp_vol_tenors)
        t_grids = np.linspace(t_min, t_max, NT, endpoint=True)
        if i == 0:
            t_grids = np.insert(t_grids, (1, 2), (dt*0.5, dt*1.5))
            NT += 2
        print('t_min: {}, t_max: {}, dt: {}, NT: {}.'.format(t_min, t_max, dt, NT))

        ## Local vol initial guess
        loc_vol_guess = compute_local_vol_init_guess(k_inputs, imp_vol_inputs, T)
        print('LV Init Guess: {}'.format(loc_vol_guess))

        ## Define Rannacher scheme
        theta = np.repeat(0.5, NT+1)
        if i == 0:
            theta[1:5] = [1.0, 1.0, 1.0, 1.0]

        ## Compute BS fwd pde prices on 5 quotes
        prices_5quotes_bs = blk_sch_fwd_pde(i, k_min, k_max, k_grids, Nk, t_min, t_max, t_grids, theta, NT, tenor_mkt_data, imp_vol_para, T_prev)
        print('BS Fwd PDE Price: {}'.format(prices_5quotes_bs))
        ################
        prices_5quotes_bs_cls = black_scholes_vanilla_fwd_norm(1, imp_vol_para.x_inputs, tenor_mkt_data.T, tenor_mkt_data.r, imp_vol_para.value_inputs)
        print('BS Closing Price: {}'.format(prices_5quotes_bs_cls))
        ################

        ## Local vol pde initial condition
        if i == 0:
            init_cond_lv = InitialConditionFirstTenor()
        else:
            init_cond_lv = InitialConditionOtherTenors(k_prev, price_prev)
        if i == 1:
            price_interpolate = init_cond_lv.compute(k_grids)
            plt.plot(k_grids, price_interpolate)
            plt.title('Pillar {} Price from last pillar interpolate'.format(i))
            plt.show()

        ## Calibrate local volatility
        debug = False
        if i == 3:
            debug = False
        lv_calibrator = LocalVolCalibration(k_min, k_max, k_grids, Nk, t_min, t_max, t_grids, theta, NT, tenor_mkt_data, imp_vol_para, init_cond_lv, prices_5quotes_bs, debug)
        loc_vol_solved, prices_5quotes_lv, price_grid_lv = lv_calibrator.calibration(loc_vol_guess)
        print('LV Fwd PDE Price: {}'.format(prices_5quotes_lv))
        print('Solved LV: {}\n'.format(loc_vol_solved))

        loc_vol_all_tenors[i,:]  = loc_vol_solved
        price_lv_all_tenors[i,:] = prices_5quotes_lv
        price_bs_all_tenors[i,:] = prices_5quotes_bs
        price_cls_all_tenors[i,:] = prices_5quotes_bs_cls
        k_all_tenors[i,:] = k_inputs
        k_prev = k_grids
        price_prev = price_grid_lv
        if i == 0:
            print_to_excel(k_prev, price_prev)
        if i == 1 or i == 0:
            plt.plot(k_prev, price_prev, '.-')
            plt.title('Pillar {} Result'.format(i))
            plt.show()

    
    return loc_vol_all_tenors, price_lv_all_tenors, price_bs_all_tenors, price_cls_all_tenors, imp_vol_tenors, k_all_tenors



if __name__ == '__main__':
    loc_vol_all_tenors, price_lv_all_tenors, price_bs_all_tenors, price_cls_all_tenors, imp_vol_tenors, k_all_tenors = loc_vol_tenor_bootstrapping()

    nb_tenors = len(imp_vol_tenors)
    nb_strikes = k_all_tenors.shape[1]

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    for i in range(nb_tenors):
        xs = k_all_tenors[i,:]
        ys = np.repeat(imp_vol_tenors[i], nb_strikes)

        ax1.plot(xs, ys,  loc_vol_all_tenors[i,:], '.-')
        ax1.set_xlabel('k')
        ax1.set_ylabel('T')
        ax1.set_zlabel('LV')
        ax1.title.set_text('Loc Vol')

        ax2.plot(xs, ys,  price_bs_all_tenors[i,:], 'b.-')
        ax2.plot(xs, ys,  price_lv_all_tenors[i,:], 'g.-')
        ax2.plot(xs, ys,  price_cls_all_tenors[i,:], 'r.-')
        ax1.set_xlabel('k')
        ax1.set_ylabel('T')
        ax1.set_zlabel('Price')
        ax2.title.set_text('Price: LV(green), BS(blue), Analytic(red)')
    plt.show()

