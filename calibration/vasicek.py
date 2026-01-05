"""
Bivariate Vasicek Model Calibration

This module implements calibration of a correlated two-factor Vasicek
(Ornstein-Uhlenbeck) model for 20-year and 30-year Treasury yields.

Model:
    dr_20 = κ_20(θ_20 - r_20)dt + σ_20 dW_20
    dr_30 = κ_30(θ_30 - r_30)dt + σ_30 dW_30
    
    where dW_20 and dW_30 are correlated with correlation ρ.

The model is calibrated using Maximum Likelihood Estimation on the
discrete-time transition density, which is Gaussian for the Vasicek model.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import warnings


@dataclass
class VasicekParams:
    """Parameters for a single Vasicek process."""
    kappa: float      # Mean reversion speed
    theta: float      # Long-run mean
    sigma: float      # Volatility
    
    def __repr__(self):
        return (f"VasicekParams(κ={self.kappa:.4f}, "
                f"θ={self.theta:.4f}, σ={self.sigma:.4f})")
    
    @property
    def half_life(self) -> float:
        """Half-life of mean reversion in years."""
        if self.kappa > 0:
            return np.log(2) / self.kappa
        return np.inf


@dataclass
class BivariateVasicekParams:
    """Parameters for correlated bivariate Vasicek model."""
    params_20y: VasicekParams
    params_30y: VasicekParams
    correlation: float
    
    def __repr__(self):
        return (f"BivariateVasicekParams(\n"
                f"  20Y: {self.params_20y}\n"
                f"  30Y: {self.params_30y}\n"
                f"  ρ = {self.correlation:.4f}\n"
                f")")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for serialization."""
        return {
            'kappa_20y': self.params_20y.kappa,
            'theta_20y': self.params_20y.theta,
            'sigma_20y': self.params_20y.sigma,
            'kappa_30y': self.params_30y.kappa,
            'theta_30y': self.params_30y.theta,
            'sigma_30y': self.params_30y.sigma,
            'correlation': self.correlation
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'BivariateVasicekParams':
        """Create from flat dictionary."""
        return cls(
            params_20y=VasicekParams(
                kappa=d['kappa_20y'],
                theta=d['theta_20y'],
                sigma=d['sigma_20y']
            ),
            params_30y=VasicekParams(
                kappa=d['kappa_30y'],
                theta=d['theta_30y'],
                sigma=d['sigma_30y']
            ),
            correlation=d['correlation']
        )


def vasicek_conditional_moments(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float
) -> Tuple[float, float]:
    """
    Compute conditional mean and variance of Vasicek process.
    
    For dr = κ(θ - r)dt + σdW, the distribution of r(t+dt) given r(t) is:
        r(t+dt) ~ N(μ, σ²)
    where:
        μ = r(t)e^(-κdt) + θ(1 - e^(-κdt))
        σ² = (σ²/2κ)(1 - e^(-2κdt))
    
    Parameters
    ----------
    r0 : float
        Current rate.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-run mean.
    sigma : float
        Volatility.
    dt : float
        Time step in years.
        
    Returns
    -------
    Tuple[float, float]
        Conditional mean and variance.
    """
    exp_kappa_dt = np.exp(-kappa * dt)
    
    mean = r0 * exp_kappa_dt + theta * (1 - exp_kappa_dt)
    
    if kappa > 1e-10:
        variance = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt))
    else:
        # Limit as kappa -> 0: variance = sigma^2 * dt
        variance = sigma**2 * dt
    
    return mean, variance


def bivariate_vasicek_conditional(
    r0_20: float,
    r0_30: float,
    params: BivariateVasicekParams,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute conditional mean vector and covariance matrix.
    
    Parameters
    ----------
    r0_20 : float
        Current 20Y yield.
    r0_30 : float
        Current 30Y yield.
    params : BivariateVasicekParams
        Model parameters.
    dt : float
        Time step in years.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Mean vector (2,) and covariance matrix (2, 2).
    """
    p20 = params.params_20y
    p30 = params.params_30y
    rho = params.correlation
    
    # Individual conditional moments
    mu_20, var_20 = vasicek_conditional_moments(r0_20, p20.kappa, p20.theta, p20.sigma, dt)
    mu_30, var_30 = vasicek_conditional_moments(r0_30, p30.kappa, p30.theta, p30.sigma, dt)
    
    # Covariance between innovations
    # Cov(ΔW_20, ΔW_30) = ρ * dt
    # After integration, covariance scales with standard deviations
    std_20 = np.sqrt(var_20)
    std_30 = np.sqrt(var_30)
    cov_20_30 = rho * std_20 * std_30
    
    mean = np.array([mu_20, mu_30])
    cov = np.array([
        [var_20, cov_20_30],
        [cov_20_30, var_30]
    ])
    
    return mean, cov


def univariate_vasicek_log_likelihood(
    rates: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float
) -> float:
    """
    Compute log-likelihood for univariate Vasicek model.
    
    Parameters
    ----------
    rates : np.ndarray
        Time series of rates.
    kappa, theta, sigma : float
        Vasicek parameters.
    dt : float
        Time step in years.
        
    Returns
    -------
    float
        Log-likelihood.
    """
    if kappa <= 0 or sigma <= 0:
        return -np.inf
    
    n = len(rates) - 1
    log_lik = 0.0
    
    for i in range(n):
        r0 = rates[i]
        r1 = rates[i + 1]
        
        mean, var = vasicek_conditional_moments(r0, kappa, theta, sigma, dt)
        std = np.sqrt(var)
        
        # Normal log-likelihood
        log_lik += stats.norm.logpdf(r1, loc=mean, scale=std)
    
    return log_lik


def bivariate_vasicek_log_likelihood(
    rates_20y: np.ndarray,
    rates_30y: np.ndarray,
    params: BivariateVasicekParams,
    dt: float
) -> float:
    """
    Compute log-likelihood for bivariate Vasicek model.
    
    Parameters
    ----------
    rates_20y : np.ndarray
        Time series of 20Y yields.
    rates_30y : np.ndarray
        Time series of 30Y yields.
    params : BivariateVasicekParams
        Model parameters.
    dt : float
        Time step in years.
        
    Returns
    -------
    float
        Log-likelihood.
    """
    n = len(rates_20y) - 1
    log_lik = 0.0
    
    for i in range(n):
        r0 = np.array([rates_20y[i], rates_30y[i]])
        r1 = np.array([rates_20y[i + 1], rates_30y[i + 1]])
        
        mean, cov = bivariate_vasicek_conditional(r0[0], r0[1], params, dt)
        
        # Check positive definiteness
        if np.linalg.det(cov) <= 0:
            return -np.inf
        
        # Bivariate normal log-likelihood
        try:
            log_lik += stats.multivariate_normal.logpdf(r1, mean=mean, cov=cov)
        except:
            return -np.inf
    
    return log_lik


def calibrate_univariate_vasicek_ols(
    rates: np.ndarray,
    dt: float
) -> VasicekParams:
    """
    Calibrate Vasicek parameters using OLS (fast approximation).
    
    Uses the AR(1) representation:
        r(t+1) = a + b*r(t) + ε
    where:
        b = exp(-κ*dt)
        a = θ(1 - b)
        
    Parameters
    ----------
    rates : np.ndarray
        Time series of rates.
    dt : float
        Time step in years.
        
    Returns
    -------
    VasicekParams
        Calibrated parameters.
    """
    # OLS regression: r(t+1) = a + b*r(t)
    y = rates[1:]
    x = rates[:-1]
    
    # Add constant
    X = np.column_stack([np.ones_like(x), x])
    
    # OLS estimates
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    a, b = beta
    
    # Residuals
    residuals = y - (a + b * x)
    residual_var = np.var(residuals, ddof=2)
    
    # Convert to Vasicek parameters
    if b >= 1:
        # No mean reversion
        kappa = 0.01  # Small positive value
        theta = np.mean(rates)
    else:
        kappa = -np.log(b) / dt
        theta = a / (1 - b)
    
    # Sigma from residual variance
    # Var(r(t+1)|r(t)) = (σ²/2κ)(1 - e^(-2κdt))
    if kappa > 1e-10:
        sigma = np.sqrt(residual_var * 2 * kappa / (1 - np.exp(-2 * kappa * dt)))
    else:
        sigma = np.sqrt(residual_var / dt)
    
    return VasicekParams(kappa=kappa, theta=theta, sigma=sigma)


def calibrate_univariate_vasicek_mle(
    rates: np.ndarray,
    dt: float,
    initial_guess: Optional[VasicekParams] = None
) -> Tuple[VasicekParams, Dict[str, Any]]:
    """
    Calibrate Vasicek parameters using MLE.
    
    Parameters
    ----------
    rates : np.ndarray
        Time series of rates.
    dt : float
        Time step in years.
    initial_guess : VasicekParams, optional
        Initial parameter guess. If None, uses OLS estimate.
        
    Returns
    -------
    Tuple[VasicekParams, Dict]
        Calibrated parameters and optimization diagnostics.
    """
    # Get initial guess from OLS if not provided
    if initial_guess is None:
        initial_guess = calibrate_univariate_vasicek_ols(rates, dt)
    
    x0 = [initial_guess.kappa, initial_guess.theta, initial_guess.sigma]
    
    def neg_log_lik(params):
        kappa, theta, sigma = params
        return -univariate_vasicek_log_likelihood(rates, kappa, theta, sigma, dt)
    
    # Bounds: kappa > 0, sigma > 0, theta unrestricted
    bounds = [(1e-6, 10), (None, None), (1e-6, None)]
    
    result = optimize.minimize(
        neg_log_lik,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    params = VasicekParams(
        kappa=result.x[0],
        theta=result.x[1],
        sigma=result.x[2]
    )
    
    diagnostics = {
        'success': result.success,
        'message': result.message,
        'log_likelihood': -result.fun,
        'n_iterations': result.nit,
        'initial_guess': initial_guess
    }
    
    return params, diagnostics


def calibrate_bivariate_vasicek(
    rates_20y: np.ndarray,
    rates_30y: np.ndarray,
    dt: float,
    use_mle: bool = True
) -> Tuple[BivariateVasicekParams, Dict[str, Any]]:
    """
    Calibrate bivariate Vasicek model.
    
    Calibrates each series independently first, then estimates correlation
    from residuals. Optionally refines with joint MLE.
    
    Parameters
    ----------
    rates_20y : np.ndarray
        Time series of 20Y yields.
    rates_30y : np.ndarray
        Time series of 30Y yields.
    dt : float
        Time step in years.
    use_mle : bool
        Whether to use MLE (True) or OLS approximation (False).
        
    Returns
    -------
    Tuple[BivariateVasicekParams, Dict]
        Calibrated parameters and diagnostics.
    """
    diagnostics = {}
    
    # Step 1: Calibrate each series independently
    if use_mle:
        params_20y, diag_20y = calibrate_univariate_vasicek_mle(rates_20y, dt)
        params_30y, diag_30y = calibrate_univariate_vasicek_mle(rates_30y, dt)
        diagnostics['20y'] = diag_20y
        diagnostics['30y'] = diag_30y
    else:
        params_20y = calibrate_univariate_vasicek_ols(rates_20y, dt)
        params_30y = calibrate_univariate_vasicek_ols(rates_30y, dt)
    
    # Step 2: Estimate correlation from residuals
    # Compute standardized residuals for each series
    n = len(rates_20y) - 1
    residuals_20y = np.zeros(n)
    residuals_30y = np.zeros(n)
    
    for i in range(n):
        # 20Y residuals
        mean_20, var_20 = vasicek_conditional_moments(
            rates_20y[i], params_20y.kappa, params_20y.theta, params_20y.sigma, dt
        )
        residuals_20y[i] = (rates_20y[i + 1] - mean_20) / np.sqrt(var_20)
        
        # 30Y residuals
        mean_30, var_30 = vasicek_conditional_moments(
            rates_30y[i], params_30y.kappa, params_30y.theta, params_30y.sigma, dt
        )
        residuals_30y[i] = (rates_30y[i + 1] - mean_30) / np.sqrt(var_30)
    
    # Correlation of standardized residuals
    correlation = np.corrcoef(residuals_20y, residuals_30y)[0, 1]
    
    diagnostics['correlation_from_residuals'] = correlation
    diagnostics['n_observations'] = n
    
    # Create bivariate params
    params = BivariateVasicekParams(
        params_20y=params_20y,
        params_30y=params_30y,
        correlation=correlation
    )
    
    # Step 3: Optional joint refinement via MLE
    if use_mle:
        # Refine correlation with joint likelihood
        def neg_log_lik_corr(rho):
            test_params = BivariateVasicekParams(
                params_20y=params_20y,
                params_30y=params_30y,
                correlation=rho[0]
            )
            return -bivariate_vasicek_log_likelihood(
                rates_20y, rates_30y, test_params, dt
            )
        
        result = optimize.minimize(
            neg_log_lik_corr,
            [correlation],
            method='L-BFGS-B',
            bounds=[(-0.999, 0.999)]
        )
        
        if result.success:
            params = BivariateVasicekParams(
                params_20y=params_20y,
                params_30y=params_30y,
                correlation=result.x[0]
            )
            diagnostics['correlation_refined'] = result.x[0]
    
    # Compute final log-likelihood
    final_ll = bivariate_vasicek_log_likelihood(rates_20y, rates_30y, params, dt)
    diagnostics['log_likelihood'] = final_ll
    
    # AIC and BIC
    n_params = 7  # 3 per series + 1 correlation
    diagnostics['aic'] = 2 * n_params - 2 * final_ll
    diagnostics['bic'] = n_params * np.log(n) - 2 * final_ll
    
    return params, diagnostics


def print_calibration_report(
    params: BivariateVasicekParams,
    diagnostics: Dict[str, Any]
) -> None:
    """Print a formatted calibration report."""
    print("=" * 60)
    print("BIVARIATE VASICEK CALIBRATION RESULTS")
    print("=" * 60)
    
    print("\n20-Year Yield Parameters:")
    p20 = params.params_20y
    print(f"  κ (mean reversion speed): {p20.kappa:.4f}")
    print(f"  θ (long-run mean):        {p20.theta:.4f} ({p20.theta*100:.2f}%)")
    print(f"  σ (volatility):           {p20.sigma:.4f} ({p20.sigma*100:.2f}%)")
    print(f"  Half-life:                {p20.half_life:.2f} years")
    
    print("\n30-Year Yield Parameters:")
    p30 = params.params_30y
    print(f"  κ (mean reversion speed): {p30.kappa:.4f}")
    print(f"  θ (long-run mean):        {p30.theta:.4f} ({p30.theta*100:.2f}%)")
    print(f"  σ (volatility):           {p30.sigma:.4f} ({p30.sigma*100:.2f}%)")
    print(f"  Half-life:                {p30.half_life:.2f} years")
    
    print(f"\nCorrelation: {params.correlation:.4f}")
    
    print("\nDiagnostics:")
    print(f"  Observations:      {diagnostics.get('n_observations', 'N/A')}")
    print(f"  Log-likelihood:    {diagnostics.get('log_likelihood', 'N/A'):.2f}")
    print(f"  AIC:               {diagnostics.get('aic', 'N/A'):.2f}")
    print(f"  BIC:               {diagnostics.get('bic', 'N/A'):.2f}")
    
    print("=" * 60)


def validate_calibration(
    rates_20y: np.ndarray,
    rates_30y: np.ndarray,
    params: BivariateVasicekParams,
    dt: float,
    n_simulations: int = 1000
) -> Dict[str, Any]:
    """
    Validate calibration by comparing simulated to historical statistics.
    
    Parameters
    ----------
    rates_20y, rates_30y : np.ndarray
        Historical rate series.
    params : BivariateVasicekParams
        Calibrated parameters.
    dt : float
        Time step.
    n_simulations : int
        Number of simulation paths for validation.
        
    Returns
    -------
    Dict
        Validation statistics.
    """
    from .rate_simulator import simulate_bivariate_vasicek
    
    n_steps = len(rates_20y)
    r0_20 = rates_20y[0]
    r0_30 = rates_30y[0]
    
    # Simulate paths
    np.random.seed(42)
    sim_20y, sim_30y = simulate_bivariate_vasicek(
        params, r0_20, r0_30, n_steps, dt, n_simulations
    )
    
    # Compare statistics
    validation = {
        'historical': {
            'mean_20y': np.mean(rates_20y),
            'std_20y': np.std(rates_20y),
            'mean_30y': np.mean(rates_30y),
            'std_30y': np.std(rates_30y),
            'correlation': np.corrcoef(rates_20y, rates_30y)[0, 1]
        },
        'simulated': {
            'mean_20y': np.mean(sim_20y),
            'std_20y': np.mean([np.std(sim_20y[:, i]) for i in range(n_simulations)]),
            'mean_30y': np.mean(sim_30y),
            'std_30y': np.mean([np.std(sim_30y[:, i]) for i in range(n_simulations)]),
            'correlation': np.mean([
                np.corrcoef(sim_20y[:, i], sim_30y[:, i])[0, 1] 
                for i in range(n_simulations)
            ])
        }
    }
    
    return validation
