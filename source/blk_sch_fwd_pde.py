import numpy as np 

from parameters import *
from initial_condition import *
from fwd_fdm import FDMCrankNicolsonNeumann


def blk_sch_fwd_pde(pillar_nb, x_min, x_max, x_values, J, t_min, t_max, t_values, theta, N, tenor_mkt_data, imp_vol_para, T_prev):

    Nb_quotes = len(imp_vol_para.x_inputs)
    prices_5quotes_bs = np.zeros_like(imp_vol_para.x_inputs)

    for i, k_quote, imp_vol_quote in zip(range(Nb_quotes), imp_vol_para.x_inputs, imp_vol_para.value_inputs):
        ## Use constant vol
        loc_vol_constant = ConstantParameter1D(imp_vol_quote)
        
        ## Define initial condition
        if pillar_nb == 0:
            init_cond_bs = InitialConditionFirstTenor()
        else:
            init_cond_bs = InitialConditionBlackScholes(i, T_prev, tenor_mkt_data, imp_vol_para)
        
        ## Build pde class
        fdm_bs = FDMCrankNicolsonNeumann(x_min, x_max, x_values, J, t_min, t_max, t_values, theta, N, \
                                        tenor_mkt_data, loc_vol_constant, init_cond_bs)
        
        ## PDE diffusion
        price_matrix = fdm_bs.step_march()
        price_grid = price_matrix[-1, :]
        price_interpolator = PiecewiseLinearParameter1D(x_values, price_grid)
        prices_5quotes_bs[i] = price_interpolator.interpolate(k_quote)
    
    return prices_5quotes_bs