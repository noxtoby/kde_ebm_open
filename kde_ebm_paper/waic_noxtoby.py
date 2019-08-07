#!/usr/bin/env python3

import numpy as np

def calculate_waic(log_lik):
    """
    Calculate Watanabe-Akaike information criterion (WAIC)
    
    Based on R code in:
    Vehtari & Gelman (2014). WAIC and cross-validation in Stan
    
    Neil Oxtoby, UCL, 2016
    """
    # Add scaling factor to handle very small log_lik, where exp(log_lik) -> 0
    # This happens log_lik < -745:
    #   exp(-745) = 4.9407e-324
    #   exp(-746) = 0
    skale = abs(np.median(log_lik))

    lpd = skale*np.log(np.mean(np.exp(log_lik/skale))) # log pointwise predictive density
    p_waic = np.var(log_lik) # estimated effective number of parameters
    elpd_waic = lpd - p_waic # expected lpd for new dataset
    waic = -2*elpd_waic

    #S = len(log_lik) #(S,n) = log_lik.shape
    #loo_weights_raw = np.exp(- (log_lik/skale - np.max(log_lik/skale)))
    #loo_weights_normalized = np.divide( loo_weights_raw , np.mean(loo_weights_raw,axis=1).reshape(-1,1) )
    #loo_weights_regularized = np.min([loo_weights_normalized,np.sqrt(S)])
    #elpd_loo = skale*np.log( np.mean(np.exp(log_lik/skale)*loo_weights_regularized ) / np.mean(loo_weights_regularized) )
    #p_loo = lpd - elpd_loo

    return waic, lpd, p_waic, elpd_waic #, p_loo, elpd_loo
