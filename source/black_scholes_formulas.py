import numpy as np
from scipy.stats import norm
from scipy.optimize import newton


def black_scholes_vanilla(S, K, T, rd, rf, sigma):
    d1 = (np.log(S/K) + (rd - rf) * T + (0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    callPrem = Nd1 * S * np.exp(- rf * T) - Nd2 * K * np.exp(- rd * T)
    return callPrem

def black_scholes_vanilla_spot_delta(S, K, T, rd, rf, sigma):
    d1 = (np.log(S/K) + (rd - rf) * T + (0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    DFf = np.exp(-rf * T)
    result = norm.cdf(d1) * DFf
    return result

def black_scholes_vanilla_fwd_delta(S, K, T, rd, rf, sigma):
    d1 = (np.log(S/K) + (rd - rf) * T + (0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    DFd = np.exp(-rd * T)
    result = norm.cdf(d1) * DFd
    return result

def black_scholes_vanilla_dual_delta(S, K, T, rd, rf, sigma):
    d1 = (np.log(S/K) + (rd - rf) * T + (0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    DFd = np.exp(-rd * T)
    result = - norm.cdf(d2) * DFd
    return result

def black_scholes_vanilla_dual_gamma(S, K, T, rd, rf, sigma):
    d1 = (np.log(S/K) + (rd - rf) * T + (0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    DFd = np.exp(-rd * T)
    result = DFd * norm.pdf(d2) / (K * sigma * np.sqrt(T))
    return result

def black_scholes_vanilla_solve_vol(S, K, T, rd, rf, vol_guess, price):
    len_price = len(price)
    if len_price==1:
        solve_func = lambda sigma: black_scholes_vanilla(S, K, T, rd, rf, sigma) - price
        result = newton(solve_func, vol_guess)
    else:
        result = np.zeros_like(price)
        for i in range(len_price):

            solve_func = lambda sigma: black_scholes_vanilla(S, K[i], T, rd, rf, sigma) - price[i]
            result[i] = newton(solve_func, vol_guess)
            print('i =', i, K[i], price[i], result[i] )
    return result
