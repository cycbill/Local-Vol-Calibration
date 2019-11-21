import numpy as np 

def compute_sum_sqr_vol_T(imp_vol_quotes, imp_vol_tenors):
    vol_max = np.amax(imp_vol_quotes, axis=1)
    T_diff = np.diff(imp_vol_tenors)

    sqr_vol_T = np.zeros_like(imp_vol_tenors)

    sqr_vol_T[1:] = vol_max[1:]**2 * T_diff
    sqr_vol_T[0] = vol_max[0]**2 * imp_vol_tenors[0]

    sum_sqr_vol_T = np.add.accumulate(sqr_vol_T)
    return sum_sqr_vol_T