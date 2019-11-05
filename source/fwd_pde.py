import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from option2 import CalibrationBasketVanillaOption
    
class ForwardPDEFirstTenor():
    def __init__(self, _option):
        self.option = _option
        
    def boundary_right(self, t, x):
        return x - self.option.strike * np.exp(-(self.option.r * (self.option.T - t)))
    
    def init_cond(self, x):
        return self.option.payoff_by_logmoney(x)