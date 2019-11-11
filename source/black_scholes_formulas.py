import numpy as np
from scipy.stats import norm
from scipy.optimize import newton, bisect, brent


def black_scholes_vanilla(S, K, T, rd, rf, vol):
    d1 = (np.log(S/K) + (rd - rf) * T + (0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    callPrem = Nd1 * S * np.exp(- rf * T) - Nd2 * K * np.exp(- rd * T)
    return callPrem

def black_scholes_vanilla_delta(S, K, T, rd, rf, vol, delta_type):
    d1 = (np.log(S/K) + (rd - rf) * T + (0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    result = norm.cdf(d1)
    if delta_type == 'spot':
        DFf = np.exp(-rf * T)
        result = result * DFf
    elif delta_type == 'fwd':
        DFd = np.exp(-rd * T)
        result = result * DFd
    return result

def black_scholes_vanilla_dual_delta(S, K, T, rd, rf, vol):
    d1 = (np.log(S/K) + (rd - rf) * T + (0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    DFd = np.exp(-rd * T)
    result = - norm.cdf(d2) * DFd
    return result

def black_scholes_vanilla_dual_gamma(S, K, T, rd, rf, vol):
    d1 = (np.log(S/K) + (rd - rf) * T + (0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    DFd = np.exp(-rd * T)
    result = DFd * norm.pdf(d2) / (K * vol * np.sqrt(T))
    return result

def black_scholes_vanilla_vega(S, K, T, rd, rf, vol):
    d1 = (np.log(S/K) + (rd - rf) * T + (0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    nd1 = norm.pdf(d1)
    DFf = np.exp(-rf * T)
    result = K * DFf * np.sqrt(T) * nd1
    return result

def black_scholes_vanilla_solve_vol(S, K, T, rd, rf, vol_guess, price):
    m = len(price)
    if m==1:
        solve_func = lambda vol: ( black_scholes_vanilla(S, K, T, rd, rf, vol) - price ) / price
        #result = newton(solve_func, vol_guess)
        result = bisect(solve_func, vol_guess*0.2, vol_guess*2.0 )
    else:
        result = np.zeros_like(price)
        for i in range(m):
            #if i==118:
            #    print('i =',i, S, K[i], T, rd, rf, vol_guess[i], price[i])
            solve_func = lambda vol: ( black_scholes_vanilla(S, K[i], T, rd, rf, vol) - price[i] ) / price[i]
            #result[i] = newton(solve_func, vol_guess[i])
            result[i] = bisect(solve_func, vol_guess[i]*0.2, vol_guess[i]*2.0 )
            #if i > 180:
            #    print('i =', i, ' K=',K[i], ' price=',price[i], ' vol=',result[i] )
    return result

def black_scholes_vanilla_solve_strike(S, strike_guess, T, rd, rf, vol, delta, delta_type):
    m = len(delta)
    if m == 1:
        solve_func = lambda strike: black_scholes_vanilla_delta(S, strike, T, rd, rf, vol, delta_type) - delta
        #result = newton(solve_func, strike_guess)
        result = bisect(solve_func, 1e-7, strike_guess*5 )
    else:
        result = np.zeros_like(delta)
        for i in range(m):
            solve_func = lambda strike: black_scholes_vanilla_delta(S, strike, T, rd, rf, vol[i], delta_type) - delta[i]
            #result[i] = newton(solve_func, strike_guess[i])
            result[i] = bisect(solve_func, 1e-7, strike_guess[i]*5 )
            print('i=', i, ' delta=', delta[i], ' strike=', result[i])
    return result