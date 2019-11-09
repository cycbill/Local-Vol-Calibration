#############################################################
# Calibrate local vol from calibration baskets on 1 pillar
# Input: 5 point quotes (k, lv)
# Process:
# (1) Fwd PDE with local vol, return k and lv series on last t grid
# (2) Interpolate on last t grid to get loc vol prices on 5 quoted k
# (3) Fwd PDE with constant implied vol, return k and lv series on last t grid
# (4) Interpolate on last t grid to get prices on 1 quoted k
# (5) Do (3), (4) for 5 times to get 5 bs prices
# (6) Set up cost function, initial loc vol guess, solve with BFGS.
# Output: 5 point quotes (k, lv)

import numpy as np 
from scipy.optimi