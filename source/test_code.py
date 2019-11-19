import numpy as np 
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt

from parameters import *


a = np.array([1, 2, 3, 4, 5])
b = np.tile(a, (10,1))

print(b)