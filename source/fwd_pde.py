import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from option2 import CalibrationBasketVanillaOption
    
class ForwardPDE():
    def __init__(self, _option):
        self.option = _option
        
    def diff_coeff(self, t, x):
        loc_vol = self.option.loc_vol
        return 0.5 * (loc_vol.interpolate(x)**2) * (x**2)
    
    def conv_coeff(self, t, x):
        loc_vol = self.option.loc_vol
        return - 0.5 * (loc_vol.interpolate(x)**2) * (x**2)
    
    def zero_coeff(self, t, x):
        return 0
    
    def source_coeff(self, t, x):
        return 0
    
    def boundary_left(self, t, x):
        return 0
        
    def boundary_right(self, t, x):
        return x - self.option.strike * np.exp(-(self.option.r * (self.option.T - t)))
    
    def init_cond(self, x):
        return self.option.payoff_by_logmoney(x)