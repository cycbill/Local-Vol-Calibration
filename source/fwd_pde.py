import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from option import VanillaOption
    
class ForwardPDE():
    def __init__(self, _option):
        self.option = _option
        
    def diff_coeff(self, t, x):
        vol = self.option.sigma
        return 0.5 * (vol**2) * (x**2)
    
    def conv_coeff(self, t, x):
        return self.option.r * x
    
    def zero_coeff(self, t, x):
        return -(self.option.r)
    
    def source_coeff(self, t, x):
        return 0
    
    def boundary_left(self, t, x):
        return 0
        
    def boundary_right(self, t, x):
        return x - self.option.strike * np.exp(-(self.option.r * (self.option.T - t)))
    
    def init_cond(self, x):
        return self.option.payoff.compute(x)