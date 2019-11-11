#####################################
# Compute the loc vol extreme quotes: k(0), k(M+1) and lv(0), lv(M+1)
#####################################

import numpy as np 
import matplotlib.pyplot as plt

from parameters import PiecewiseLinearParameter1D

class LocalVolatility():
    def __init__(self, _k_quotes, _loc_vol_quotes, _imp_vol_atm, _T):
        self.x_inputs = _k_quotes
        self.value_inputs = _loc_vol_quotes
        self.imp_vol_atm = _imp_vol_atm
        self.T = _T
        self.m = len(self.x_inputs)
        self.x_extend, self.value_extend = self.compute_lv_extreme_quotes()
        self.loc_vol_para = PiecewiseLinearParameter1D(self.x_extend, self.value_extend)
    

    def compute_lv_extreme_quotes(self):
        lv_min = 0.0
        lv_max = 3.0 * max(self.value_inputs)
        
        k = np.zeros(self.m + 2)
        k[1:-1] = self.x_inputs
        k[0] = min( -3.0 * self.imp_vol_atm * np.sqrt(self.T), k[1] - 1e-7)
        k[-1] = max( 3.0 * self.imp_vol_atm * np.sqrt(self.T), k[-2] + 1e-7)

        lv = np.zeros(self.m + 2)
        lv[1:-1] = self.value_inputs
        lv[0] = min( max( lv[1] +  (lv[2] - lv[1]) / (k[2] - k[1]) * (k[0] - k[1]), \
                        lv_min), lv_max)
        lv[-1] = min( max( lv[-2] + (lv[-2] - lv[-3]) / (k[-2] - k[-3]) * (k[-1] - k[-2]), \
                        lv_min), lv_max)
        
        if lv[0] == lv_max:
            k[0] = k[1] + (lv_max - lv[1]) * (k[2] - k[1]) / (lv[2] - lv[1])
        if lv[-1] == lv_max:
            k[-1] = k[-2] + (lv_max - lv[-2]) * (k[-2] - k[-3]) / (lv[-2] - lv[-3])

        return k, lv


    def interpolate(self, x):
        return self.loc_vol_para.interpolate(x)



if __name__ == '__main__':
    k_quotes = np.array([-0.2, -0.1, 0, 0.1, 0.2])
    loc_vol_quotes = np.array([0.2, 0.15, 0.12, 0.15, 0.22])
    imp_vol_atm = 0.10
    T = 1

    k, lv = compute_lv_extreme_quotes(k_quotes, loc_vol_quotes, imp_vol_atm, T)

    loc_vol_para = LocalVolatility(k_quotes, loc_vol_quotes, imp_vol_atm, T)
    
    print(loc_vol_para.k_extend)
    print(loc_vol_para.lv_extend)
    plt.plot(k, lv, 'r.')

    k_grid = np.linspace(-1.0, 1.0, 201, endpoint=True)
    lv_grid = loc_vol_para.interpolate(k_grid)
    plt.plot(k_grid, lv_grid, 'b-')
    plt.title('Interpolation')
    plt.show()
    
