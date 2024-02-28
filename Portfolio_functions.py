# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:13:59 2024

@author: ktsar
"""

import numpy as np

def portfolio_var(returns, weights):
    """
    Calculate portfolio variance.

    Parameters:
    returns (array-like): Array or DataFrame of historical returns for each asset.
    weights (array-like): Array of portfolio weights for each asset.

    Returns:
    float: Portfolio variance.
    """
    cov_matrix = np.cov(returns, rowvar=False)
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    return port_var

def portfolio_var_at_level(returns, weights, alpha=0.05):
    """
    Calculate portfolio Value at Risk (VaR) using Variance-Covariance method.

    Parameters:
    returns (array-like): Array or DataFrame of historical returns for each asset.
    weights (array-like): Array of portfolio weights for each asset.
    alpha (float): Confidence level, e.g., 0.05 for 5%.

    Returns:
    float: Portfolio Value at Risk.
    """
    port_var = portfolio_var(returns, weights)
    z_score = np.sqrt(port_var) * np.percentile(np.random.normal(0, 1, 10000), 100 * (1 - alpha))
    var = np.dot(weights.T, np.dot(np.cov(returns, rowvar=False), weights)) ** 0.5 * z_score
    return var