import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fdm import FDMEulerExplicit
from pde import BlackScholesPDE
from option import VanillaOption
from payoff import PayOffCall, PayOffPut

K = 0.5
r = 0.25
v = 0.2
T = 1.00

x_dom = 3.0
J = 51
t_dom = T
N = 50

pay_off_call = PayOffCall(K)
call_option = VanillaOption(K, r, T, v, pay_off_call)
bs_pde = BlackScholesPDE(call_option)
fdm_euler = FDMEulerExplicit(x_dom, J, t_dom, N, bs_pde)

prices = fdm_euler.step_march()

plt.plot(prices)
plt.show()