import numpy as np
import pandas as pd
from scipy.sparse import diags
import matplotlib.pyplot as plt
from fwd_pde import ForwardPDE

class FDMCrankNicolsonNeumann():
    def __init__(self, _x_min, _x_dom, _J, _t_dom, _N, _pde):
        self.x_min = _x_min
        self.x_dom = _x_dom
        self.J = _J
        self.t_dom = _t_dom
        self.N = _N
        self.pde = _pde
        self.calculate_step_size()
        self.set_initial_conditions()
        
    def calculate_step_size(self):
        self.dx = (self.x_dom - self.x_min) / (self.J - 1)
        self.dt = self.t_dom / (self.N - 1)
        
    def set_initial_conditions(self):
        self.x_values = np.linspace(self.x_min, self.x_dom, self.J, endpoint=True)
        self.old_result = self.pde.init_cond(self.x_values)
        self.new_result = np.zeros_like(self.old_result)
        self.prev_t = 0
        self.cur_t = 0
        plt.plot(self.x_values, self.old_result, color=(self.cur_t * 0.9, 0.2, 0.5), linestyle='--')
        
    def calculate_boundary_conditions(self):
        self.new_result[0] = 2 * self.new_result[1] - self.new_result[2]
        self.new_result[-1] = 2 * self.new_result[-2] - self.new_result[-3]
        
    def calculate_inner_domain(self):
        zeta = self.pde.conv_coeff(self.prev_t, self.x_values[1:-1]) / (self.dx**2)

        alpha = - zeta * (1 + 0.5 * self.dx)
        beta = 2 * zeta
        gamma = - zeta * (1 - 0.5 * self.dx)

        #A = diags([alpha, beta, gamma], [-1, 0, 1]).toarray()
        A = np.zeros((self.J, self.J))
        for i in range(1, self.J-1):
            A[i, i-1] = alpha[i-1]
            A[i, i] = beta[i-1]
            A[i, i+1] = gamma[i-1]
        

        #A = np.concatenate((np.zeros(self.J), A), axis=0)
        #A = np.concatenate(A, (np.zeros(self.J)), axis=0)

        left_coeff_matrix = 1 - 0.5 * self.dt * A

        left_coeff_matrix[0,0] = 1
        left_coeff_matrix[0,1] = -2
        left_coeff_matrix[0,2] = 1

        left_coeff_matrix[-1,-1] = 1
        left_coeff_matrix[-1,-2] = -2
        left_coeff_matrix[-1,-3] = 1

        right_coeff_matrix = 1 + 0.5 * self.dt * A
        right_result = right_coeff_matrix * self.old_result
        self.new_result = np.linalg.solve(left_coeff_matrix, right_result)

        '''
        dt_diffu = self.dt * (self.pde.diff_coeff(self.prev_t, self.x_values[1:-1]))
        dtdxp2_conv = self.dt * self.dx * 0.5 * (self.pde.conv_coeff(self.prev_t, self.x_values[1:-1]))
        
        alpha = 0.5 * (dt_diffu - dtdxp2_conv) / (self.dx ** 2)
        beta = -dt_diffu / (self.dx ** 2) + 0.5 * self.dt * self.pde.zero_coeff(self.prev_t, self.x_values[1:-1])
        gamma = 0.5 * (dt_diffu + dtdxp2_conv) / (self.dx ** 2)
        
        A = alpha
        B = 1 + beta
        C = gamma
        D = - alpha * self.old_result[:-2] + (1 - beta) * self.old_result[1:-1] - gamma * self.old_result[2:]

        B[0] = 2 * alpha[0] + 1 + beta[0]
        C[0] = - alpha[0] + gamma[0]
        D[0] = (- 2 * alpha[0] + 1 - beta[0]) * self.old_result[1] + (alpha[0] - gamma[0]) * self.old_result[2]
            
        A[-1] = alpha[-1] - gamma[-1]
        B[-1] = 1 + beta[-1] + 2 * gamma[-1]
        D[-1] = (- alpha[-1] + gamma[-1]) * self.old_result[-3] + (1 - beta[-1] - 2 * gamma[-1]) * self.old_result[-2]

        coeff_matrix = diags([A[1:], B, C[:-1]], [-1, 0, 1], shape=(self.J - 2, self.J - 2)).toarray()
        solved_vector = np.linalg.solve(coeff_matrix, D)
        self.new_result[1:-1] = solved_vector
        '''

    def step_march(self):
        for n in range(1, self.N):
            self.cur_t = n * self.dt
            self.calculate_inner_domain()
            self.calculate_boundary_conditions()
            self.old_result = self.new_result
            if n < 2:
                plt.plot(self.x_values, self.old_result, color=(self.cur_t * 0.9, 0.2, 0.5))
            self.prev_t = self.cur_t
        return self.old_result, self.x_values

