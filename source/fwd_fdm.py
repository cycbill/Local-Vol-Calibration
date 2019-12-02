import numpy as np
import pandas as pd
from scipy.sparse import diags
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class FDMCrankNicolsonNeumann():
    def __init__(self, _x_min, _x_max, _x_values, _J, _t_min, _t_max, _t_values, _theta, _N, _tenor_mkt_data, _vol_data, _init_condition):
        self.x_min = _x_min
        self.x_max = _x_max
        self.x_values = _x_values
        self.J = int(_J)
        self.t_min = _t_min
        self.t_max = _t_max
        self.t_grids = _t_values
        self.theta = _theta
        self.N = int(_N)
        self.tenor_mkt_data = _tenor_mkt_data
        self.vol_data = _vol_data
        self.init_condition = _init_condition
        self.set_initial_conditions()

        
    def set_initial_conditions(self):
        self.dx = (self.x_max - self.x_min) / (self.J - 1)        # J is number of strike step intervals
        self.result = np.zeros((self.N, self.J))
        self.result[0,:] = self.init_condition.compute(self.x_values)
        #self.old_result = self.init_condition.compute(self.x_values)
        #self.new_result = np.zeros_like(self.old_result)
        self.zeta = self.vol_data.interpolate(self.x_values) ** 2 / (2 * self.dx ** 2)

        ## Pre-compute coefficient matrices M_L and M_R, which are time independent.
        self.Matrix_A = np.zeros((self.J, self.J))
        for j in range(1, self.J-1):
            self.Matrix_A[j, j-1] = (1 + self.dx / 2) * self.zeta[j]
            self.Matrix_A[j, j] =  -2 * self.zeta[j]
            self.Matrix_A[j, j+1] = (1 - self.dx / 2) * self.zeta[j]
        self.Matrix_I = np.identity(self.J)
        self.Matrix_I[0,0] = 0
        self.Matrix_I[-1, -1] = 0


    def calculate_inner_domain(self, n):
        self.dt = self.cur_t - self.prev_t        # N is number of maturity step intervals

        ## Construct left and right matrices
        self.Matrix_L = self.Matrix_I - self.theta[n] * self.dt * self.Matrix_A
        self.Matrix_R = self.Matrix_I + (1 - self.theta[n]) * self.dt * self.Matrix_A
        
        ## Null gamma condition
        #null_gamma_cond = np.array([1 - self.dx / 2, -2, 1 + self.dx / 2])
        #self.Matrix_L[0, :3] = null_gamma_cond
        #self.Matrix_L[self.J, self.J-2:] = null_gamma_cond

        ## Dichilet condition
        self.Matrix_L[0, 0] = 1
        self.Matrix_L[-1, -1] = 1

        right_vector = np.matmul(self.Matrix_R, self.result[n-1, :])

        ## Dichilet condition
        right_vector[0] = self.init_condition.compute(self.x_values[0])
        right_vector[-1] = self.init_condition.compute(self.x_values[-1])

        self.result[n, :] = np.linalg.solve(self.Matrix_L, right_vector)

    def step_march(self):
        for n in range(1, self.N):
            self.prev_t = self.t_grids[n-1]
            self.cur_t = self.t_grids[n]
            self.calculate_inner_domain(n)
        return self.result

