import numpy as np 

class StrikeGridsAllTenors():
    def __init__(self, _nb_tenors):
        self.k_min = np.zeros(_nb_tenors)
        self.k_max = np.zeros(_nb_tenors)
        self.dk = np.zeros(_nb_tenors)
        self.Nk = np.zeros(_nb_tenors)

    def strike_grid_discretization(self, tenor_i, k_min_extrplt, k_max_extrplt, imp_vol_para, tenor_mkt_data, sum_sqr_vol_T):
        #_k_min = max(k_min_extrplt, -5 * np.sqrt(sum_sqr_vol_T))
        #_k_max = min(k_max_extrplt,  5 * np.sqrt(sum_sqr_vol_T))
        _k_min = k_min_extrplt
        _k_max = k_max_extrplt

        vol_max = max(imp_vol_para.value_inputs)
        _k_min = min(_k_min, imp_vol_para.x_inputs[0] - 0.75 * vol_max * np.sqrt(tenor_mkt_data.T))
        _k_max = max(_k_max, imp_vol_para.x_inputs[-1] + 0.75 * vol_max * np.sqrt(tenor_mkt_data.T))
        self.k_min[tenor_i] = _k_min
        self.k_max[tenor_i] = _k_max

        if tenor_i > 0:
            k_minus = min(self.k_min[tenor_i - 1], -3 * vol_max * np.sqrt(tenor_mkt_data.T))
            k_plus  = max(self.k_max[tenor_i - 1],  3 * vol_max * np.sqrt(tenor_mkt_data.T))
        else:
            k_minus = -3 * vol_max * np.sqrt(tenor_mkt_data.T)
            k_plus  =  3 * vol_max * np.sqrt(tenor_mkt_data.T)

        Nk_tmp = min( max( (k_plus - k_minus)/(2.00e-2), 51), 201)
        dk = (k_plus - k_minus) / (Nk_tmp - 1)
        #self.Nk[tenor_i] = min( max( np.ceil( (_k_max - _k_min)/dk ), 51), 201)
        self.Nk[tenor_i] = min( max( (_k_max - _k_min)/dk, 51), 201)
        self.dk[tenor_i] = (_k_max - _k_min) / (self.Nk[tenor_i] - 1)
        return self.k_min[tenor_i], self.k_max[tenor_i], self.dk[tenor_i], self.Nk[tenor_i]