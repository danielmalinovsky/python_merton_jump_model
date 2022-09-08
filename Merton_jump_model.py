# %% 
#Libraries
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar   
import time
from scipy.optimize import minimize

# %%
# User defined functions

def merton_jump_func(S, T, r, sigma, lambd, norm_mean, norm_volatility, steps, N_paths):
    dt = T/steps
    poisson_process = np.multiply(np.random.poisson(lam*dt, size = (steps, N_paths)),
                                    np.random.normal(norm_mean, norm_volatility, size= (steps, N_paths))).cumsum(axis=0)

    geometric_brownian_motion = np.cumsum(
        (r - (sigma ** 2) / 2) - lambd*(norm_mean + (norm_volatility**2*0.5))*dt + sigma*np.sqrt(dt)*np.random.normal(size=(steps, N_paths)), axis=0
    )

    return np.exp(geometric_brownian_motion + poisson_process) * S

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S *norm.cdf(d1) - K * np.exp(-r*T)* norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def merton_jump_call(S, K, T, r, sigma, norm_mean, norm_volatility, lambd):
    p = 0
    for i in range(40):
        r_k = r - lambd*(norm_mean-1) + (i*np.log(norm_mean)) / T
        sigma_k = np.sqrt(sigma**2 + (i * norm_volatility**2) / T)
        k_fact = np.math.factorial(i)
        p = p + ((np.exp(-norm_mean*lambd*T) * (norm_mean*lambd*T)**i / (k_fact) * black_scholes_call(S, K, T, r_k, sigma_k)))
    return p

def merton_jump_put(S, K, T, r, sigma, norm_mean, norm_volatility, lambd):
    p = 0
    for i in range(40):
        r_k = r - lambd*(norm_mean-1) + (i*np.log(norm_mean)) / T
        sigma_k = np.sqrt(sigma**2 + (i * norm_volatility**2) / T)
        k_fact = np.math.factorial(i)
        p = p + ((np.exp(-norm_mean*lambd*T) * (norm_mean*lambd*T)**i / (k_fact) * black_scholes_put(S, K, T, r_k, sigma_k)))
    return p

def implied_vol(opt_value, S, K, T, r, type_='call'):
    
    def call_obj(sigma):
        return abs(black_scholes_call(S, K, T, r, sigma) - opt_value)
    
    def put_obj(sigma):
        return abs(black_scholes_put(S, K, T, r, sigma) - opt_value)
    
    if type_ == 'call':
        res = minimize_scalar(call_obj, bounds=(0.01,6), method='bounded')
        return res.x
    elif type_ == 'put':
        res = minimize_scalar(put_obj, bounds=(0.01,6),
                              method='bounded')
        return res.x
    else:
        raise ValueError("type_ must be 'put' or 'call'")

def optimalisation (x, market_prices, strikes):
    candidate_prices = merton_jump_call(S, strikes, T, r, sigma=x[0], norm_mean=x[1], norm_volatility=x[2], lambd=x[3])
    return np.linalg.norm(market_prices - candidate_prices, 2)

# %%
# Merton Jump Diffusion process simulation

S = 100 # current stock price
T = 1 # time to maturity
r = 0.02 # risk free rate
m = 0 # meean of jump size
v = 0.3 # standard deviation of jump
lam =1 # intensity of jump i.e. number of jumps per annum
steps =100 # time steps
Npaths = 100 # number of paths to simulate
sigma = 0.2 # annaul standard deviation , for weiner process

merton_jump_paths = merton_jump_func(S, T, r, sigma, lam, m, v, steps, Npaths)

plt.plot(merton_jump_paths)
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Merton Jump Diffusion Paths. \n' + str(Npaths) + ' Paths each with ' + str(steps) + ' time steps')

# %%
# 
K = 100

np.random.seed(15)

mcprice = np.maximum(merton_jump_paths[-1]-K,0).mean() * np.exp(-r*T)
cf_price = merton_jump_call(S, K, T, r, sigma, np.exp(m+v**2*0.5), v, lam)

print('Merton Price =', cf_price)
print('Monte Carlo Merton Price =', mcprice)
print('Black Scholes Price =', black_scholes_call(S,K,T,r, sigma))

# %%
# Mertons Jump Diffusion volatility smile

#mean value has to be set different from zero for marton_jump_call/put function to work
m = 1


strikes = np.arange(50,150,1)

merton_jump_prices = merton_jump_call(S, strikes, T, r, sigma, m, v, lam)
merton_jump_implied_volatilities = [implied_vol(c, S, k, T, r) for c,k in zip(merton_jump_prices, strikes)]

plt.plot(strikes, merton_jump_implied_volatilities, label='Implied Volatility Smile')
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.axvline(S, color='black', linestyle='dashed', linewidth=2,label="Spot")
plt.title('Merton Jump Diffusion Volatility Smile')
plt.legend()

# %%
# Model Calibration

option_data = df = pd.read_csv('https://raw.githubusercontent.com/codearmo/data/master/calls_calib_example.csv')

T = option_data['T'].values[0]
S = option_data.F.values[0]
r = 0
x0 = [0.15, 1, 0.1, 1]
bounds = ((0.1, np.inf), (0.01, 2), (1e-5, np.inf), (0, 5))
strikes = option_data.Strike.values
prices = option_data.Midpoint.values

result = minimize(optimalisation, method='SLSQP', x0=x0, args=(prices, strikes), bounds=bounds, tol=1e-20, options={"maxiter":1000})

sigma_t = result.x[0]
norm_mean_t = result.x[1]
norm_volatility_t = result.x[2]
lambd_t = result.x[3]

print('Calibrated Volatlity = ', sigma_t)
print('Calibrated Jump Mean = ', norm_mean_t)
print('Calibrated Jump Std = ', norm_volatility_t)
print('Calibrated intensity = ', lambd_t)


# %%

option_data['least_sq_V'] = merton_jump_call(S, option_data.Strike, option_data['T'], 0, sigma_t, norm_mean_t, norm_volatility_t, lambd_t)

plt.scatter(df.Strike, df.Midpoint,label= 'Observed Prices')
plt.plot(df.Strike, df.least_sq_V, color='black',label= 'Fitted Prices')
plt.legend()
plt.xlabel('Strike')
plt.ylabel('Value in $')
plt.title('Merton Model Optimal Params')
# %%
