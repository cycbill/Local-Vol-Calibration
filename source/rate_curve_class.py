#####################################
# Build the rate curve class
#####################################

import numpy as np 

from parameters import PiecewiseLinearParameter1D

class RateCurve(PiecewiseLinearParameter1D):
    def __init__(self, _T, _r):
        PiecewiseLinearParameter1D.__init__(self, _T, _r)

    def DF(self, t0, t1):
        result = 1 / self.CF(t0, t1)
        return result

    def CF(self, t0, t1):
        zc0 = self.interpolate(t0)
        zc1 = self.interpolate(t1)
        cf0 = np.exp(zc0 * t0)
        cf1 = np.exp(zc1 * t1)
        result = cf1 / cf0
        #result = np.exp( self.integral(t0, t1) )
        return result