import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs_fdm import FDMEulerExplicit
from bs_pde import BlackScholesPDE
from option import VanillaOption
from payoff import PayOffCall, PayOffPut
from black_scholes_formulas import black_scholes_vanilla

K = 0.5
r = 0.25
v = 0.2
T = 1.00

x_dom = 1.0
J = 21
t_dom = T
N = 21

pay_off_call = PayOffCall(K)
call_option = VanillaOption(K, r, T, v, pay_off_call)
bs_pde = BlackScholesPDE(call_option)
fdm_euler = FDMEulerExplicit(x_dom, J, t_dom, N, bs_pde)

prices, x_values = fdm_euler.step_march()

plt.plot(x_values, prices)
plt.show()


bs_price_by_spot = lambda spot: black_scholes_vanilla(spot, K, T, r, 0, v)

bs_closing_prices = bs_price_by_spot(x_values)
plt.plot(x_values, bs_closing_prices, 'r')
plt.plot(x_values, prices, 'b--')
plt.show()

## assume spot is 0.6
S = 0.6
F = S * np.exp(r * T)   # 0.7704152500126448
print("Forward: ", F)

k = np.log(K/F)
print("log moneyness: ", k)