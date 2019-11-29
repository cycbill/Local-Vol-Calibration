import numpy as np
import pandas as pd
from scipy.sparse import diags
import matplotlib.pyplot as plt


class FDMCrankNicolsonNeumann():
    def __init__(self, _x_min, _x_max, _J, _t_min, _t_max, _N, _tenor_mkt_data, _vol_data, _init_condition):
        self.x_min = _x_min
        self.x_max = _x_max
        self.J = _J
        self.t_min = _t_min
        self.t_max = _t_max
        self.N = _N
        self.tenor_mkt_data = _tenor_mkt_data
        self.vol_data = _vol_data
        self.init_condition = _init_condition
        self.calculate_step_size()
        self.set_initial_conditions()
        
    def calculate_step_size(self):
        self.dx = (self.x_max - self.x_min) / self.J        # J is number of strike step intervals
        self.dt = (self.t_max - self.t_min) / self.N        # N is number of maturity step intervals
        
    def set_initial_conditions(self):
        self.x_values = np.linspace(self.x_min, self.x_max, self.J+1, endpoint=True)
        self.old_result = self.init_condition.compute(self.x_values)
        self.new_result = np.zeros_like(self.old_result)
        self.prev_t = 0
        self.cur_t = 0
        self.zeta = self.vol_data.interpolate(self.x_values) ** 2 / (4 * self.dx ** 2)

        ## Pre-compute coefficient matrices M_L and M_R, which are time independent.
        Matrix_A = np.zeros((self.J+1, self.J+1))
        for j in range(1, self.J):
            Matrix_A[j, j-1] = (1 + self.dx / 2) * self.zeta[j]
            Matrix_A[j, j] =  -2 * self.zeta[j]
            Matrix_A[j, j+1] = (1 - self.dx / 2) * self.zeta[j]
        Matrix_I = np.identity(self.J+1)
        Matrix_I[0,0] = 0
        Matrix_I[self.J, self.J] = 0
        self.Matrix_L = Matrix_I - self.dt * Matrix_A
        self.Matrix_R = Matrix_I + self.dt * Matrix_A
        
        ## Null gamma condition
        #null_gamma_cond = np.array([1 - self.dx / 2, -2, 1 + self.dx / 2])
        #self.Matrix_L[0, :3] = null_gamma_cond
        #self.Matrix_L[self.J, self.J-2:] = null_gamma_cond

        ## Dichilet condition
        self.Matrix_L[0, 0] = 1
        self.Matrix_L[-1, -1] = 1

        #plt.plot(self.x_values, self.old_result, color=(self.cur_t * 0.2, 0.9, 0.5), linestyle='--')
        K = self.tenor_mkt_data.fwd * np.exp(self.x_values)
        plt.plot(K, self.old_result * self.tenor_mkt_data.spot, color=(self.cur_t * 0.2, 0.9, 0.5), linestyle='--')


    def calculate_inner_domain(self):
        right_vector = np.matmul(self.Matrix_R, self.old_result)

        ## Dichilet condition
        right_vector[0] = self.init_condition.compute(self.x_values[0])
        right_vector[-1] = self.init_condition.compute(self.x_values[-1])

        self.new_result = np.linalg.solve(self.Matrix_L, right_vector)

    def step_march(self):
        for n in range(1, self.N):
            self.cur_t = n * self.dt
            self.calculate_inner_domain()
            self.old_result = self.new_result
            if n != 0:
                #print(n)
                #print(self.old_result)
                #plt.plot(self.x_values, self.old_result, color=(self.cur_t * 0.9, 0.2, 0.5))
                K = self.tenor_mkt_data.fwd * np.exp(self.x_values)
                plt.plot(K, self.old_result * self.tenor_mkt_data.spot, color=(self.cur_t * 0.9, 0.2, 0.5))
            self.prev_t = self.cur_t
        return self.old_result, self.x_values

