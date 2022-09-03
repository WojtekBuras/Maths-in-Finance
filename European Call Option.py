#!/usr/bin/env python
# coding: utf-8

# In[91]:


from numpy import random
from scipy.stats import norm
import math

# Pricing European Call Option through simulation
# Stochastic Equation for the Price: dS = r S dt + Vol S dW
# Initial Price: S(0) = S_0, 
# Volatility: Vol = const, 
# Risk free rate: r = const,
# Maturity: T
# Brownian Motion: W (dW follows Normal Distribution with mean = 0, volatility = 1)
# Option Return at t = T: max( 0, S(T) - K ), where K is a Strike.
# We must discount the Return by the factor of exp( -r * T ) to get the Value at t = 0

# We will price the Option running N random walks described by the stochastic equation and taking the average Value
# at T

# We would like to compare the results for different N and compare the results with the Black - Scholes Formula:
# V = S(0) N(d1) - K exp (-r T ) N(d2), where N(x) is the Gaussian Integral from - inf to x. 

def simulate_path( S_0, r, vol, T ):
    # 1 step per day, maturity given in years
    steps = T * 365
    step_vol = vol / math.sqrt( 365 )
    step_r = r / 365
    
    S = S_0
    
    for step in range( steps ):
        dS = step_r * S + step_vol * S * random.normal()   
        S += dS
    
    return S 



def expected_value( N, S_0, r, vol, T, K):
    path_lst = [];
    
    for i in range(N):
        payment = simulate_path( S_0, r, vol, T ) - K
        value = payment * math.exp( -r * T )
        path_lst.append( max( 0, value ) ) 

    return round( sum(path_lst) / N, 7 ) 


    
def black_scholes( S_0, r, vol, T, K ):
    d1 = ( math.log( S_0 / K ) + ( r + vol**2 / 2 ) * T ) / ( vol * math.sqrt( T ) )
    
    d2 = d1 - vol * math.sqrt( T )
    
    value = S_0 * norm.cdf( d1 ) - K * math.exp( -r * T ) * norm.cdf( d2 )
    
    return round( value, 7 )


def compare_values_for_N( S_0, r, vol, T, K ):
    print("number of simulations | expected value ")
    
    lst = [100,1000,10000,20000,50000,100000]
     
    for i in lst:
        print(str(i) + ': ' + str ( expected_value( i, S_0, r, vol, T,  K ) ) )
        
    print("Black Scholes" + ': ' + str( black_scholes( S_0, r, vol, T, K ) ) ) 



# In[92]:


compare_values_for_N( 1, 0.02, 0.1, 2, 1 )


# In[ ]:




