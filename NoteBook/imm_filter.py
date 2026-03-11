import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import pandas as pd

def init_state(price0, Q_init,R_init, alpha_prior=10.0):

    # price0 : first price observed, we use it to initialize
    # Q and R init, are the initale noise process and mesurment noise
    #alpha prior: is the the default strengh of bars, 10 bars

    return{
        "x_hat": price0,
        "P": 1.0,
        "Q_k"        : Q_init,
        "R_k"        : R_init,
        "alpha_R"    : alpha_prior,
        "beta_R"     : alpha_prior * R_init,
        "alpha_Q"    : alpha_prior,
        "beta_Q"     : alpha_prior * Q_init,
        "sigma_innov": 1e-4,  # we start near zero noise, it is calculated after
    }

def vb_step(price_t, state):
    #values
    x_hat = state[0]
    P = state[1]
    Q_k = state[2]
    R_k = state[3]

    
  

