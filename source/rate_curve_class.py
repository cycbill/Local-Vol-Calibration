#####################################
# Build the rate curve class
#####################################

import numpy as np 

from parameters import PiecewiseLinearParameter1D

class RateCurve(PiecewiseLinearParameter1D):
    def __init__(self, _T, _r):
        PiecewiseLinearParameter1D.__init__(self, _T, _r)

    def DF(self, t0, t1):
        result = np.exp( - self.integral(t0, t1) )
        return result

    def CF(self, t0, t1):
        result = np.exp( self.integral(t0, t1) )
        return result