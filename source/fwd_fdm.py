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
        
    def calculate_boundary_conditions(self):
        #self.new_result[0] = self.pde.boundary_left(self.cur_t, self.x_values[0])
        #self.new_result[-1] = self.pde.boundary_right(self.cur_t, self.x_values[-1])
        self.new_result[0] = 2 * self.new_result[1] - self.new_result[2]
        self.new_result[-1] = 2 * self.new_result[-2] - self.new_result[-3]
        
    def calculate_inner_domain(self):
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
        

    def step_march(self):
        for n in range(1, self.N):
            self.cur_t = n * self.dt
            self.calculate_inner_domain()
            self.calculate_boundary_conditions()
            self.old_result = self.new_result
            plt.plot(self.old_result, color=(self.cur_t * 0.9, 0.2, 0.5))
            self.prev_t = self.cur_t
        return self.old_result

