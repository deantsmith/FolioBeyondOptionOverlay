"""
TLT Price Regression Model

This module calibrates the relationship between Treasury yields and TLT prices.
Uses the average of 20Y and 30Y yields as the primary driver.

The model captures:
1. The level relationship (price ~ f(yield))
2. Duration-based sensitivity (modified duration)
3. Convexity adjustment for large yield moves
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any


@dataclass
class TLTRegressionParams:
    """Parameters for TLT price model."""
    
    # Linear model: log(P) = α + β * yield
    alpha: float           # Intercept
    beta: float            # Yield sensitivity (negative for bonds)
    
    # Optional convexity term: log(P) = α + β*y + γ*y²
    gamma: Optional[float] = None
    
    # Model diagnostics
    r_squared: float = 0.0
    rmse: float = 0.0
    
    # Implied duration at reference yield
    reference_yield: float = 0.0
    implied_duration: float = 0.0
    
    def __repr__(self):
        conv_str = f", γ={self.gamma:.4f}" if self.gamma else ""
        return (f"TLTRegressionParams(α={self.alpha:.4f}, β={self.beta:.4f}"
                f"{conv_str}, R²={self.r_squared:.4f})")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        d = {
            'alpha': self.alpha,
            'beta': self.beta,
            'r_squared': self.r_squared,
            'rmse': self.rmse,
            'reference_yield': self.reference_yield,
            'implied_duration': self.implied_duration
        }
        if self.gamma is not None:
            d['gamma'] = self.gamma
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'TLTRegressionParams':
        """Create from dictionary."""
        return cls(
            alpha=d['alpha'],
            beta=d['beta'],
            gamma=d.get('gamma'),
            r_squared=d.get('r_squared', 0.0),
            rmse=d.get('rmse', 0.0),
            reference_yield=d.get('reference_yield', 0.0),
            implied_duration=d.get('implied_duration', 0.0)
        )


def calibrate_tlt_regression(
    yields: np.ndarray,
    tlt_prices: np.ndarray,
    include_convexity: bool = True
) -> Tuple[TLTRegressionParams, Dict[str, Any]]:
    """
    Calibrate TLT price model on yields.
    
    Fits: log(P) = α + β*y [+ γ*y²]
    
    Parameters
    ----------
    yields : np.ndarray
        Average yield (20Y + 30Y) / 2, in decimal form.
    tlt_prices : np.ndarray
        TLT closing prices.
    include_convexity : bool
        Whether to include the quadratic term.
        
    Returns
    -------
    Tuple[TLTRegressionParams, Dict]
        Calibrated parameters and diagnostics.
    """
    # Use log prices for better fit
    log_prices = np.log(tlt_prices)
    n = len(yields)
    
    if include_convexity:
        # Quadratic regression: log(P) = α + β*y + γ*y²
        X = np.column_stack([np.ones(n), yields, yields**2])
        beta, residuals, rank, s = np.linalg.lstsq(X, log_prices, rcond=None)
        alpha, beta_coef, gamma = beta
    else:
        # Linear regression: log(P) = α + β*y
        X = np.column_stack([np.ones(n), yields])
        beta, residuals, rank, s = np.linalg.lstsq(X, log_prices, rcond=None)
        alpha, beta_coef = beta
        gamma = None
    
    # Predictions and residuals
    if include_convexity:
        log_prices_pred = alpha + beta_coef * yields + gamma * yields**2
    else:
        log_prices_pred = alpha + beta_coef * yields
    
    resid = log_prices - log_prices_pred
    
    # R-squared
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((log_prices - np.mean(log_prices))**2)
    r_squared = 1 - ss_res / ss_tot
    
    # RMSE (in log space and price space)
    rmse_log = np.sqrt(np.mean(resid**2))
    rmse_price = np.sqrt(np.mean((tlt_prices - np.exp(log_prices_pred))**2))
    
    # Implied duration at reference yield
    reference_yield = np.mean(yields)
    # Duration = -d(log P)/dy = -β - 2γy (if convexity)
    if include_convexity:
        implied_duration = -(beta_coef + 2 * gamma * reference_yield)
    else:
        implied_duration = -beta_coef
    
    params = TLTRegressionParams(
        alpha=alpha,
        beta=beta_coef,
        gamma=gamma,
        r_squared=r_squared,
        rmse=rmse_price,
        reference_yield=reference_yield,
        implied_duration=implied_duration
    )
    
    # Additional diagnostics
    diagnostics = {
        'n_observations': n,
        'rmse_log': rmse_log,
        'rmse_price': rmse_price,
        'mean_yield': np.mean(yields),
        'std_yield': np.std(yields),
        'mean_price': np.mean(tlt_prices),
        'std_price': np.std(tlt_prices),
        'residuals': resid,
        'predictions_log': log_prices_pred,
        'predictions_price': np.exp(log_prices_pred)
    }
    
    # Statistical significance
    if include_convexity:
        # F-test for convexity term
        X_reduced = np.column_stack([np.ones(n), yields])
        beta_reduced, _, _, _ = np.linalg.lstsq(X_reduced, log_prices, rcond=None)
        log_prices_reduced = beta_reduced[0] + beta_reduced[1] * yields
        ss_res_reduced = np.sum((log_prices - log_prices_reduced)**2)
        
        f_stat = ((ss_res_reduced - ss_res) / 1) / (ss_res / (n - 3))
        p_value = 1 - stats.f.cdf(f_stat, 1, n - 3)
        
        diagnostics['convexity_f_stat'] = f_stat
        diagnostics['convexity_p_value'] = p_value
        diagnostics['convexity_significant'] = p_value < 0.05
    
    return params, diagnostics


def calibrate_return_regression(
    yield_changes: np.ndarray,
    tlt_returns: np.ndarray,
    yields: Optional[np.ndarray] = None
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Calibrate return-based model (alternative approach).
    
    Fits: r_TLT = α + β * Δy [+ γ * Δy * y]
    
    This approach directly models returns and can be more robust
    for simulation purposes.
    
    Parameters
    ----------
    yield_changes : np.ndarray
        Daily changes in average yield.
    tlt_returns : np.ndarray
        Daily TLT returns.
    yields : np.ndarray, optional
        Yield levels for state-dependent sensitivity.
        
    Returns
    -------
    Tuple[Dict, Dict]
        Model coefficients and diagnostics.
    """
    n = len(yield_changes)
    
    # Basic model: r = α + β * Δy
    X = np.column_stack([np.ones(n), yield_changes])
    beta, _, _, _ = np.linalg.lstsq(X, tlt_returns, rcond=None)
    
    predictions = beta[0] + beta[1] * yield_changes
    resid = tlt_returns - predictions
    
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((tlt_returns - np.mean(tlt_returns))**2)
    r_squared = 1 - ss_res / ss_tot
    
    coefficients = {
        'alpha': beta[0],
        'beta': beta[1],  # This should be negative and approx = -duration/100
        'implied_duration': -beta[1] * 100  # Convert to duration
    }
    
    diagnostics = {
        'r_squared': r_squared,
        'rmse': np.sqrt(np.mean(resid**2)),
        'n_observations': n,
        'residuals': resid
    }
    
    return coefficients, diagnostics


def predict_tlt_price(
    yields: np.ndarray,
    params: TLTRegressionParams
) -> np.ndarray:
    """
    Predict TLT prices from yields.
    
    Parameters
    ----------
    yields : np.ndarray
        Average yields in decimal form.
    params : TLTRegressionParams
        Calibrated model parameters.
        
    Returns
    -------
    np.ndarray
        Predicted TLT prices.
    """
    if params.gamma is not None:
        log_prices = params.alpha + params.beta * yields + params.gamma * yields**2
    else:
        log_prices = params.alpha + params.beta * yields
    
    return np.exp(log_prices)


def predict_tlt_return(
    yield_start: float,
    yield_end: float,
    params: TLTRegressionParams
) -> float:
    """
    Predict TLT return from yield change.
    
    Parameters
    ----------
    yield_start : float
        Starting yield.
    yield_end : float
        Ending yield.
    params : TLTRegressionParams
        Calibrated parameters.
        
    Returns
    -------
    float
        Predicted return.
    """
    price_start = predict_tlt_price(np.array([yield_start]), params)[0]
    price_end = predict_tlt_price(np.array([yield_end]), params)[0]
    
    return (price_end - price_start) / price_start


def compute_duration_convexity(
    yield_level: float,
    params: TLTRegressionParams
) -> Tuple[float, float]:
    """
    Compute implied duration and convexity at a given yield level.
    
    For log(P) = α + β*y + γ*y²:
        Duration = -d(log P)/dy = -(β + 2γy)
        Convexity = d²(log P)/dy² = -2γ (approximately)
    
    Parameters
    ----------
    yield_level : float
        Current yield level.
    params : TLTRegressionParams
        Model parameters.
        
    Returns
    -------
    Tuple[float, float]
        Duration and convexity.
    """
    if params.gamma is not None:
        duration = -(params.beta + 2 * params.gamma * yield_level)
        convexity = -2 * params.gamma
    else:
        duration = -params.beta
        convexity = 0.0
    
    return duration, convexity


def print_regression_report(
    params: TLTRegressionParams,
    diagnostics: Dict[str, Any]
) -> None:
    """Print a formatted regression report."""
    print("=" * 60)
    print("TLT REGRESSION CALIBRATION RESULTS")
    print("=" * 60)
    
    print("\nModel: log(TLT) = α + β × yield" + 
          (" + γ × yield²" if params.gamma else ""))
    
    print(f"\nParameters:")
    print(f"  α (intercept):     {params.alpha:.4f}")
    print(f"  β (yield coef):    {params.beta:.4f}")
    if params.gamma:
        print(f"  γ (convexity):     {params.gamma:.4f}")
    
    print(f"\nImplied Duration at {params.reference_yield*100:.2f}% yield:")
    print(f"  Duration:          {params.implied_duration:.2f} years")
    
    if params.gamma:
        _, convexity = compute_duration_convexity(params.reference_yield, params)
        print(f"  Convexity:         {convexity:.2f}")
    
    print(f"\nFit Statistics:")
    print(f"  R-squared:         {params.r_squared:.4f}")
    print(f"  RMSE (price):      ${params.rmse:.2f}")
    print(f"  RMSE (log):        {diagnostics.get('rmse_log', 0):.4f}")
    print(f"  Observations:      {diagnostics.get('n_observations', 'N/A')}")
    
    if 'convexity_significant' in diagnostics:
        sig = "Yes" if diagnostics['convexity_significant'] else "No"
        print(f"\nConvexity significant at 5%: {sig} "
              f"(p={diagnostics['convexity_p_value']:.4f})")
    
    print("=" * 60)


def validate_regression(
    yields: np.ndarray,
    tlt_prices: np.ndarray,
    params: TLTRegressionParams
) -> Dict[str, Any]:
    """
    Validate regression model with additional diagnostics.
    
    Parameters
    ----------
    yields : np.ndarray
        Historical yields.
    tlt_prices : np.ndarray
        Historical TLT prices.
    params : TLTRegressionParams
        Calibrated parameters.
        
    Returns
    -------
    Dict
        Validation statistics.
    """
    # Predictions
    pred_prices = predict_tlt_price(yields, params)
    
    # Error analysis
    errors = tlt_prices - pred_prices
    pct_errors = errors / tlt_prices * 100
    
    validation = {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_abs_error': np.max(np.abs(errors)),
        'mean_pct_error': np.mean(pct_errors),
        'std_pct_error': np.std(pct_errors),
        'max_abs_pct_error': np.max(np.abs(pct_errors))
    }
    
    # Check for autocorrelation in residuals
    residuals = np.log(tlt_prices) - np.log(pred_prices)
    autocorr_lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    validation['residual_autocorr_lag1'] = autocorr_lag1
    
    # Durbin-Watson statistic
    dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    validation['durbin_watson'] = dw
    
    return validation
