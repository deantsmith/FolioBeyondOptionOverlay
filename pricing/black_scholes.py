"""
Black-Scholes Option Pricing

This module implements Black-Scholes pricing for European options,
including Greeks calculations needed for delta-based strike selection.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class OptionQuote:
    """Container for option pricing results."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    
    # Additional info
    intrinsic: float = 0.0
    time_value: float = 0.0
    
    def __repr__(self):
        return (f"OptionQuote(price=${self.price:.2f}, Δ={self.delta:.3f}, "
                f"Γ={self.gamma:.4f}, Θ={self.theta:.3f}, ν={self.vega:.3f})")


def d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate d1 parameter for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate d2 parameter for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return d1(S, K, r, sigma, T) - sigma * np.sqrt(T)


def call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate Black-Scholes call option price.
    
    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    r : float
        Risk-free rate (annualized).
    sigma : float
        Implied volatility (annualized).
    T : float
        Time to expiration in years.
        
    Returns
    -------
    float
        Call option price.
    """
    if T <= 0:
        return max(S - K, 0)
    
    d1_val = d1(S, K, r, sigma, T)
    d2_val = d2(S, K, r, sigma, T)
    
    price = S * stats.norm.cdf(d1_val) - K * np.exp(-r * T) * stats.norm.cdf(d2_val)
    return price


def put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate Black-Scholes put option price.
    
    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    r : float
        Risk-free rate (annualized).
    sigma : float
        Implied volatility (annualized).
    T : float
        Time to expiration in years.
        
    Returns
    -------
    float
        Put option price.
    """
    if T <= 0:
        return max(K - S, 0)
    
    d1_val = d1(S, K, r, sigma, T)
    d2_val = d2(S, K, r, sigma, T)
    
    price = K * np.exp(-r * T) * stats.norm.cdf(-d2_val) - S * stats.norm.cdf(-d1_val)
    return price


def call_delta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate call option delta."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    return stats.norm.cdf(d1(S, K, r, sigma, T))


def put_delta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate put option delta."""
    if T <= 0:
        return -1.0 if S < K else 0.0
    return stats.norm.cdf(d1(S, K, r, sigma, T)) - 1


def gamma(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate option gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1_val = d1(S, K, r, sigma, T)
    return stats.norm.pdf(d1_val) / (S * sigma * np.sqrt(T))


def vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate option vega (same for calls and puts). Returns per 1% vol change."""
    if T <= 0:
        return 0.0
    d1_val = d1(S, K, r, sigma, T)
    return S * stats.norm.pdf(d1_val) * np.sqrt(T) / 100  # Per 1% vol change


def call_theta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate call theta (daily decay)."""
    if T <= 0:
        return 0.0
    d1_val = d1(S, K, r, sigma, T)
    d2_val = d2(S, K, r, sigma, T)
    
    term1 = -S * stats.norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * stats.norm.cdf(d2_val)
    
    return (term1 + term2) / 365  # Daily theta


def put_theta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate put theta (daily decay)."""
    if T <= 0:
        return 0.0
    d1_val = d1(S, K, r, sigma, T)
    d2_val = d2(S, K, r, sigma, T)
    
    term1 = -S * stats.norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * stats.norm.cdf(-d2_val)
    
    return (term1 + term2) / 365  # Daily theta


def price_call(S: float, K: float, r: float, sigma: float, T: float) -> OptionQuote:
    """
    Full call option pricing with all Greeks.
    
    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    r : float
        Risk-free rate (annualized).
    sigma : float
        Implied volatility (annualized).
    T : float
        Time to expiration in years.
        
    Returns
    -------
    OptionQuote
        Complete option quote with price and Greeks.
    """
    price = call_price(S, K, r, sigma, T)
    intrinsic = max(S - K, 0)
    
    return OptionQuote(
        price=price,
        delta=call_delta(S, K, r, sigma, T),
        gamma=gamma(S, K, r, sigma, T),
        theta=call_theta(S, K, r, sigma, T),
        vega=vega(S, K, r, sigma, T),
        intrinsic=intrinsic,
        time_value=price - intrinsic
    )


def price_put(S: float, K: float, r: float, sigma: float, T: float) -> OptionQuote:
    """
    Full put option pricing with all Greeks.
    
    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    r : float
        Risk-free rate (annualized).
    sigma : float
        Implied volatility (annualized).
    T : float
        Time to expiration in years.
        
    Returns
    -------
    OptionQuote
        Complete option quote with price and Greeks.
    """
    price = put_price(S, K, r, sigma, T)
    intrinsic = max(K - S, 0)
    
    return OptionQuote(
        price=price,
        delta=put_delta(S, K, r, sigma, T),
        gamma=gamma(S, K, r, sigma, T),
        theta=put_theta(S, K, r, sigma, T),
        vega=vega(S, K, r, sigma, T),
        intrinsic=intrinsic,
        time_value=price - intrinsic
    )


def find_strike_by_delta(
    S: float,
    r: float,
    sigma: float,
    T: float,
    target_delta: float,
    option_type: str = 'put',
    tolerance: float = 0.001
) -> float:
    """
    Find strike price that produces target delta.
    
    Parameters
    ----------
    S : float
        Current underlying price.
    r : float
        Risk-free rate.
    sigma : float
        Implied volatility.
    T : float
        Time to expiration in years.
    target_delta : float
        Target delta (absolute value, e.g., 0.20 for 20-delta).
    option_type : str
        'put' or 'call'.
    tolerance : float
        Acceptable delta error.
        
    Returns
    -------
    float
        Strike price with target delta.
    """
    # For puts, delta is negative, so we work with absolute value
    if option_type == 'put':
        target_delta = -abs(target_delta)
        delta_func = lambda K: put_delta(S, K, r, sigma, T)
    else:
        target_delta = abs(target_delta)
        delta_func = lambda K: call_delta(S, K, r, sigma, T)
    
    # Binary search for strike
    if option_type == 'put':
        # Put strikes are below spot for OTM
        K_low = S * 0.5
        K_high = S * 1.0
    else:
        # Call strikes are above spot for OTM
        K_low = S * 1.0
        K_high = S * 1.5
    
    for _ in range(50):  # Max iterations
        K_mid = (K_low + K_high) / 2
        delta_mid = delta_func(K_mid)
        
        if abs(delta_mid - target_delta) < tolerance:
            return K_mid
        
        if option_type == 'put':
            # For puts: higher strike = more negative delta
            if delta_mid > target_delta:
                K_low = K_mid
            else:
                K_high = K_mid
        else:
            # For calls: higher strike = lower delta
            if delta_mid > target_delta:
                K_low = K_mid
            else:
                K_high = K_mid
    
    return K_mid


def implied_volatility(
    option_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = 'call',
    tolerance: float = 0.0001,
    max_iterations: int = 100
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Parameters
    ----------
    option_price : float
        Market price of the option.
    S : float
        Current underlying price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    T : float
        Time to expiration in years.
    option_type : str
        'call' or 'put'.
    tolerance : float
        Convergence tolerance.
    max_iterations : int
        Maximum iterations.
        
    Returns
    -------
    float
        Implied volatility.
    """
    if T <= 0:
        return 0.0
    
    # Initial guess
    sigma = 0.20
    
    price_func = call_price if option_type == 'call' else put_price
    
    for _ in range(max_iterations):
        price = price_func(S, K, r, sigma, T)
        vega_val = vega(S, K, r, sigma, T) * 100  # Convert back from per-1%
        
        if vega_val < 1e-10:
            break
        
        diff = option_price - price
        
        if abs(diff) < tolerance:
            return sigma
        
        sigma = sigma + diff / vega_val
        sigma = max(0.01, min(sigma, 5.0))  # Keep sigma in reasonable range
    
    return sigma


# Vectorized versions for efficiency with simulation paths

def call_price_vec(
    S: np.ndarray,
    K: float,
    r: float,
    sigma: np.ndarray,
    T: float
) -> np.ndarray:
    """Vectorized call pricing for arrays of spot prices and vols."""
    if T <= 0:
        return np.maximum(S - K, 0)
    
    d1_val = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2_val = d1_val - sigma * np.sqrt(T)
    
    return S * stats.norm.cdf(d1_val) - K * np.exp(-r * T) * stats.norm.cdf(d2_val)


def put_price_vec(
    S: np.ndarray,
    K: float,
    r: float,
    sigma: np.ndarray,
    T: float
) -> np.ndarray:
    """Vectorized put pricing for arrays of spot prices and vols."""
    if T <= 0:
        return np.maximum(K - S, 0)
    
    d1_val = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2_val = d1_val - sigma * np.sqrt(T)
    
    return K * np.exp(-r * T) * stats.norm.cdf(-d2_val) - S * stats.norm.cdf(-d1_val)
