"""
Volatility Estimation and IV Modeling

This module handles:
1. Realized volatility calculation
2. IV-price relationship calibration
3. IV estimation for simulation
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List


@dataclass
class VolatilityParams:
    """Parameters for volatility model."""
    
    # Base realized volatility (annualized)
    base_vol: float
    
    # IV-price relationship: IV = base_iv + sensitivity * (price_pct_from_ref)
    # When TLT drops, IV typically rises
    base_iv: float
    price_sensitivity: float  # Negative: IV rises when price falls
    
    # Reference price for sensitivity calculation
    reference_price: float
    
    # Volatility regime indicators
    vol_percentile_25: float = 0.0
    vol_percentile_50: float = 0.0
    vol_percentile_75: float = 0.0
    
    def __repr__(self):
        return (f"VolatilityParams(base_vol={self.base_vol:.2%}, "
                f"base_iv={self.base_iv:.2%}, "
                f"sensitivity={self.price_sensitivity:.4f})")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'base_vol': self.base_vol,
            'base_iv': self.base_iv,
            'price_sensitivity': self.price_sensitivity,
            'reference_price': self.reference_price,
            'vol_percentile_25': self.vol_percentile_25,
            'vol_percentile_50': self.vol_percentile_50,
            'vol_percentile_75': self.vol_percentile_75
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'VolatilityParams':
        """Create from dictionary."""
        return cls(**d)


def compute_realized_volatility(
    prices: np.ndarray,
    window: int = 21,
    annualization: float = 252.0,
    method: str = 'close_to_close'
) -> np.ndarray:
    """
    Compute rolling realized volatility.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series.
    window : int
        Rolling window in trading days.
    annualization : float
        Trading days per year for annualization.
    method : str
        Volatility estimation method:
        - 'close_to_close': Standard deviation of log returns
        - 'parkinson': High-low range estimator (if OHLC available)
        
    Returns
    -------
    np.ndarray
        Rolling annualized volatility (same length as prices, with NaN for initial window).
    """
    log_returns = np.diff(np.log(prices))
    
    # Pad to match original length
    log_returns = np.concatenate([[np.nan], log_returns])
    
    # Rolling standard deviation
    realized_vol = pd.Series(log_returns).rolling(window).std().values
    
    # Annualize
    realized_vol *= np.sqrt(annualization)
    
    return realized_vol


def compute_volatility_statistics(
    prices: np.ndarray,
    window: int = 21,
    annualization: float = 252.0
) -> Dict[str, float]:
    """
    Compute comprehensive volatility statistics.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series.
    window : int
        Rolling window.
    annualization : float
        Trading days per year.
        
    Returns
    -------
    Dict
        Volatility statistics.
    """
    # Full-sample volatility (always computable if we have at least 2 prices)
    log_returns = np.diff(np.log(prices))
    full_sample_vol = np.std(log_returns) * np.sqrt(annualization) if len(log_returns) > 1 else 0.15
    
    # Rolling realized vol
    realized_vol = compute_realized_volatility(prices, window, annualization)
    valid_vol = realized_vol[~np.isnan(realized_vol)]
    
    # Handle case where we don't have enough data for rolling vol
    if len(valid_vol) == 0:
        # Fall back to full-sample vol for all statistics
        stats_dict = {
            'full_sample_vol': full_sample_vol,
            'rolling_vol_mean': full_sample_vol,
            'rolling_vol_std': 0.0,
            'rolling_vol_min': full_sample_vol,
            'rolling_vol_max': full_sample_vol,
            'rolling_vol_p25': full_sample_vol * 0.9,
            'rolling_vol_p50': full_sample_vol,
            'rolling_vol_p75': full_sample_vol * 1.1,
            'rolling_vol_current': full_sample_vol,
            'warning': f'Insufficient data for rolling vol (need > {window} observations)'
        }
    else:
        stats_dict = {
            'full_sample_vol': full_sample_vol,
            'rolling_vol_mean': np.mean(valid_vol),
            'rolling_vol_std': np.std(valid_vol),
            'rolling_vol_min': np.min(valid_vol),
            'rolling_vol_max': np.max(valid_vol),
            'rolling_vol_p25': np.percentile(valid_vol, 25),
            'rolling_vol_p50': np.percentile(valid_vol, 50),
            'rolling_vol_p75': np.percentile(valid_vol, 75),
            'rolling_vol_current': valid_vol[-1]
        }
    
    return stats_dict


def calibrate_iv_price_relationship(
    prices: np.ndarray,
    window: int = 21,
    annualization: float = 252.0
) -> Tuple[VolatilityParams, Dict[str, Any]]:
    """
    Calibrate the IV-price relationship.
    
    Fits: IV ≈ base_iv + sensitivity * (P - P_ref) / P_ref
    
    This captures the empirical observation that IV rises when 
    bond prices fall (yields spike).
    
    Parameters
    ----------
    prices : np.ndarray
        TLT price series.
    window : int
        Rolling window for vol calculation.
    annualization : float
        Trading days per year.
        
    Returns
    -------
    Tuple[VolatilityParams, Dict]
        Calibrated parameters and diagnostics.
    """
    # Get volatility statistics first
    vol_stats = compute_volatility_statistics(prices, window, annualization)
    
    # Reference price: median of the sample
    reference_price = np.median(prices)
    
    # Compute realized vol as proxy for IV
    realized_vol = compute_realized_volatility(prices, window, annualization)
    
    # Remove NaN values
    valid_mask = ~np.isnan(realized_vol)
    valid_prices = prices[valid_mask]
    valid_vol = realized_vol[valid_mask]
    
    # Handle case with insufficient data for regression
    if len(valid_vol) < 10:
        # Use defaults based on full-sample vol
        base_iv = vol_stats['full_sample_vol']
        price_sensitivity = -0.02  # Default: 1% price drop → 0.02% IV increase
        r_squared = 0.0
        
        diagnostics = {
            'r_squared': r_squared,
            'n_observations': len(valid_vol),
            'correlation_price_vol': 0.0,
            'vol_stats': vol_stats,
            'residuals': np.array([]),
            'warning': f'Insufficient data for IV-price regression (n={len(valid_vol)}). Using defaults.'
        }
    else:
        # Price deviation from reference
        price_deviation = (valid_prices - reference_price) / reference_price
        
        # Regress vol on price deviation
        # vol = a + b * price_deviation
        X = np.column_stack([np.ones(len(price_deviation)), price_deviation])
        beta, _, _, _ = np.linalg.lstsq(X, valid_vol, rcond=None)
        
        base_iv = beta[0]
        price_sensitivity = beta[1]  # Should be negative
        
        # Predictions and R-squared
        vol_pred = base_iv + price_sensitivity * price_deviation
        ss_res = np.sum((valid_vol - vol_pred)**2)
        ss_tot = np.sum((valid_vol - np.mean(valid_vol))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        diagnostics = {
            'r_squared': r_squared,
            'n_observations': len(valid_vol),
            'correlation_price_vol': np.corrcoef(valid_prices, valid_vol)[0, 1] if len(valid_vol) > 1 else 0.0,
            'vol_stats': vol_stats,
            'residuals': valid_vol - vol_pred
        }
    
    params = VolatilityParams(
        base_vol=vol_stats['full_sample_vol'],
        base_iv=base_iv,
        price_sensitivity=price_sensitivity,
        reference_price=reference_price,
        vol_percentile_25=vol_stats['rolling_vol_p25'],
        vol_percentile_50=vol_stats['rolling_vol_p50'],
        vol_percentile_75=vol_stats['rolling_vol_p75']
    )
    
    return params, diagnostics


def estimate_iv(
    current_price: float,
    params: VolatilityParams,
    vol_regime: str = 'normal'
) -> float:
    """
    Estimate implied volatility for a given price.
    
    Parameters
    ----------
    current_price : float
        Current TLT price.
    params : VolatilityParams
        Calibrated volatility parameters.
    vol_regime : str
        Volatility regime adjustment:
        - 'low': Use 25th percentile
        - 'normal': Use base IV
        - 'high': Use 75th percentile
        - 'stressed': Add 50% to base IV
        
    Returns
    -------
    float
        Estimated implied volatility (annualized).
    """
    # Price-dependent adjustment
    price_deviation = (current_price - params.reference_price) / params.reference_price
    iv = params.base_iv + params.price_sensitivity * price_deviation
    
    # Regime adjustment
    if vol_regime == 'low':
        iv = min(iv, params.vol_percentile_25)
    elif vol_regime == 'high':
        iv = max(iv, params.vol_percentile_75)
    elif vol_regime == 'stressed':
        iv *= 1.5
    
    # Floor IV at a minimum reasonable level
    iv = max(iv, 0.05)  # At least 5%
    
    return iv


def estimate_iv_path(
    price_path: np.ndarray,
    params: VolatilityParams
) -> np.ndarray:
    """
    Estimate IV along a simulated price path.
    
    Parameters
    ----------
    price_path : np.ndarray
        Simulated TLT prices.
    params : VolatilityParams
        Calibrated parameters.
        
    Returns
    -------
    np.ndarray
        Estimated IV at each point.
    """
    price_deviation = (price_path - params.reference_price) / params.reference_price
    iv = params.base_iv + params.price_sensitivity * price_deviation
    
    # Floor
    iv = np.maximum(iv, 0.05)
    
    return iv


def compute_term_structure_adjustment(
    base_iv: float,
    dte: int,
    mean_reversion_days: float = 60.0
) -> float:
    """
    Adjust IV for term structure.
    
    Short-dated options typically have higher IV during stress,
    but lower IV in calm markets. This applies a simple mean-reversion
    adjustment.
    
    Parameters
    ----------
    base_iv : float
        Base implied volatility estimate.
    dte : int
        Days to expiration.
    mean_reversion_days : float
        Half-life of vol mean reversion in days.
        
    Returns
    -------
    float
        Term-structure adjusted IV.
    """
    # Simple exponential mean reversion to long-term average
    # For short DTE, IV is more extreme (higher when above average, lower when below)
    long_term_iv = 0.15  # Assume 15% long-term average
    
    decay_factor = np.exp(-dte / mean_reversion_days)
    
    # Blend: more weight to current IV for short DTE
    adjusted_iv = base_iv * decay_factor + long_term_iv * (1 - decay_factor)
    
    return adjusted_iv


def print_volatility_report(
    params: VolatilityParams,
    diagnostics: Dict[str, Any]
) -> None:
    """Print formatted volatility report."""
    print("=" * 60)
    print("VOLATILITY MODEL CALIBRATION RESULTS")
    print("=" * 60)
    
    # Check for warnings
    if 'warning' in diagnostics:
        print(f"\n⚠️  {diagnostics['warning']}")
    
    print("\nBase Volatility Estimates:")
    print(f"  Full-sample realized vol: {params.base_vol:.2%}")
    print(f"  Base IV (at reference):   {params.base_iv:.2%}")
    print(f"  Reference price:          ${params.reference_price:.2f}")
    
    print("\nIV-Price Relationship:")
    print(f"  Sensitivity:              {params.price_sensitivity:.4f}")
    if params.price_sensitivity != 0:
        print(f"  Interpretation:           1% price drop → "
              f"{-params.price_sensitivity*0.01*100:.2f}% IV increase")
    
    print("\nVolatility Regime Percentiles:")
    print(f"  25th percentile (low):    {params.vol_percentile_25:.2%}")
    print(f"  50th percentile (median): {params.vol_percentile_50:.2%}")
    print(f"  75th percentile (high):   {params.vol_percentile_75:.2%}")
    
    print(f"\nModel Fit:")
    print(f"  R-squared:                {diagnostics['r_squared']:.4f}")
    print(f"  Price-vol correlation:    {diagnostics['correlation_price_vol']:.4f}")
    print(f"  Observations:             {diagnostics['n_observations']}")
    
    # Example IV estimates
    print("\nExample IV Estimates:")
    for pct_change in [-10, -5, 0, 5, 10]:
        test_price = params.reference_price * (1 + pct_change/100)
        test_iv = estimate_iv(test_price, params)
        print(f"  Price {pct_change:+d}% (${test_price:.2f}): IV = {test_iv:.2%}")
    
    print("=" * 60)


def analyze_volatility_regimes(
    prices: np.ndarray,
    window: int = 21,
    annualization: float = 252.0,
    n_regimes: int = 3
) -> Dict[str, Any]:
    """
    Analyze volatility regimes in historical data.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series.
    window : int
        Rolling window.
    annualization : float
        Days per year.
    n_regimes : int
        Number of regimes to identify.
        
    Returns
    -------
    Dict
        Regime analysis results.
    """
    realized_vol = compute_realized_volatility(prices, window, annualization)
    valid_mask = ~np.isnan(realized_vol)
    valid_vol = realized_vol[valid_mask]
    
    # Regime thresholds based on percentiles
    if n_regimes == 3:
        thresholds = [33, 67]
        regime_names = ['Low', 'Normal', 'High']
    elif n_regimes == 2:
        thresholds = [50]
        regime_names = ['Low', 'High']
    else:
        thresholds = np.linspace(0, 100, n_regimes + 1)[1:-1].tolist()
        regime_names = [f'Regime {i+1}' for i in range(n_regimes)]
    
    percentile_values = [np.percentile(valid_vol, p) for p in thresholds]
    
    # Assign regimes
    regimes = np.zeros(len(valid_vol), dtype=int)
    for i, threshold in enumerate(percentile_values):
        regimes[valid_vol > threshold] = i + 1
    
    # Regime statistics
    regime_stats = {}
    for i, name in enumerate(regime_names):
        mask = regimes == i
        regime_stats[name] = {
            'count': np.sum(mask),
            'pct_time': np.mean(mask) * 100,
            'mean_vol': np.mean(valid_vol[mask]),
            'std_vol': np.std(valid_vol[mask])
        }
    
    return {
        'regime_names': regime_names,
        'thresholds': percentile_values,
        'regime_stats': regime_stats,
        'regimes': regimes
    }
