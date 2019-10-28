import numpy as np
from scipy.stats import norm


def black_scholes_vanilla(S, K, T, rd, rf, sigma):
    d1 = (np.log(S/K) + (rd - rf) * T + (0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    callPrem = Nd1 * S * np.exp(- rf * T) - Nd2 * K * np.exp(- rd * T)
    return callPrem