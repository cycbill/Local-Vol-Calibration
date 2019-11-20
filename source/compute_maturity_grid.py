import numpy as np 

def compute_maturity_grid(i, imp_vol_tenors):
    t_length = 0
    if i==0:
        t_min = 0
        t_max = imp_vol_tenors[i]
    else:
        t_min = imp_vol_tenors[i - 1]
        t_max = imp_vol_tenors[i]
    t_length = t_max - t_min
    NT = max(int(50 * t_length), 10)
    dt = t_length / NT
    return t_min, t_max, dt, NT