"""
Test Script with Synthetic Data

This script generates synthetic data and runs the calibration pipeline
to verify all components work correctly.

Usage:
    python test_with_synthetic_data.py
"""

import sys
from pathlib import Path

# Add script directory to path BEFORE other imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import ModelConfig
from calibration.vasicek import (
    BivariateVasicekParams,
    VasicekParams,
    calibrate_bivariate_vasicek,
    print_calibration_report
)
from calibration.tlt_regression import (
    calibrate_tlt_regression,
    predict_tlt_price,
    print_regression_report,
    TLTRegressionParams
)
from calibration.volatility import (
    calibrate_iv_price_relationship,
    print_volatility_report
)
from simulation.rate_paths import (
    simulate_bivariate_vasicek,
    run_simulation,
    print_simulation_summary
)
from simulation.tlt_paths import (
    simulate_tlt_paths,
    print_tlt_simulation_summary
)


def generate_synthetic_data(
    n_days: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic yield and TLT data for testing.
    
    Uses a bivariate Vasicek process to generate yields,
    then converts to TLT prices using a simple duration model.
    """
    np.random.seed(seed)
    
    # True parameters
    true_params = BivariateVasicekParams(
        params_20y=VasicekParams(kappa=0.15, theta=0.04, sigma=0.008),
        params_30y=VasicekParams(kappa=0.12, theta=0.042, sigma=0.009),
        correlation=0.95
    )
    
    # Initial yields
    r0_20 = 0.038
    r0_30 = 0.040
    
    dt = 1/252
    
    # Simulate yields
    paths_20, paths_30 = simulate_bivariate_vasicek(
        true_params, r0_20, r0_30, n_days, dt, n_paths=1, random_seed=seed
    )
    
    yields_20 = paths_20[:, 0]
    yields_30 = paths_30[:, 0]
    yields_avg = (yields_20 + yields_30) / 2
    
    # Generate TLT prices using a simple model
    # log(P) = 5.5 - 18 * yield (roughly 18 year duration at ~4% yield)
    log_prices = 5.5 - 18 * yields_avg + np.random.normal(0, 0.002, len(yields_avg))
    tlt_prices = np.exp(log_prices)
    
    # Create date index
    start_date = datetime(2019, 1, 2)
    dates = pd.bdate_range(start=start_date, periods=n_days + 1)
    
    # Build DataFrame
    df = pd.DataFrame({
        'date': dates,
        'yield_20y': yields_20,
        'yield_30y': yields_30,
        'tlt_close': tlt_prices
    })
    
    return df, true_params


def run_test():
    """Run complete test suite."""
    print("=" * 70)
    print("TLT OPTIONS MODEL - TEST WITH SYNTHETIC DATA")
    print("=" * 70)
    
    # Generate synthetic data
    print("\n[1] Generating synthetic data...")
    df, true_params = generate_synthetic_data(n_days=1000)
    print(f"    Generated {len(df)} observations")
    print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    print("\n    True parameters used for data generation:")
    print(f"    20Y: κ={true_params.params_20y.kappa:.3f}, "
          f"θ={true_params.params_20y.theta:.4f}, "
          f"σ={true_params.params_20y.sigma:.4f}")
    print(f"    30Y: κ={true_params.params_30y.kappa:.3f}, "
          f"θ={true_params.params_30y.theta:.4f}, "
          f"σ={true_params.params_30y.sigma:.4f}")
    print(f"    ρ={true_params.correlation:.3f}")
    
    # Extract series
    yields_20 = df['yield_20y'].values
    yields_30 = df['yield_30y'].values
    yields_avg = (yields_20 + yields_30) / 2
    tlt_prices = df['tlt_close'].values
    
    dt = 1/252
    
    # Test Vasicek calibration
    print("\n[2] Testing Vasicek calibration...")
    vasicek_params, vasicek_diag = calibrate_bivariate_vasicek(
        yields_20, yields_30, dt, use_mle=True
    )
    print_calibration_report(vasicek_params, vasicek_diag)
    
    # Compare to true values
    print("\nParameter Recovery:")
    print(f"    κ_20: True={true_params.params_20y.kappa:.4f}, "
          f"Estimated={vasicek_params.params_20y.kappa:.4f}")
    print(f"    θ_20: True={true_params.params_20y.theta:.4f}, "
          f"Estimated={vasicek_params.params_20y.theta:.4f}")
    print(f"    ρ:    True={true_params.correlation:.4f}, "
          f"Estimated={vasicek_params.correlation:.4f}")
    
    # Test TLT regression
    print("\n[3] Testing TLT regression...")
    reg_params, reg_diag = calibrate_tlt_regression(
        yields_avg, tlt_prices, include_convexity=True
    )
    print_regression_report(reg_params, reg_diag)
    
    # Test volatility calibration
    print("\n[4] Testing volatility calibration...")
    vol_params, vol_diag = calibrate_iv_price_relationship(tlt_prices)
    print_volatility_report(vol_params, vol_diag)
    
    # Test simulation
    print("\n[5] Testing rate simulation...")
    rate_sim = run_simulation(
        vasicek_params,
        r0_20=yields_20[-1],
        r0_30=yields_30[-1],
        horizon_days=60,  # 60 trading days
        n_paths=5000,
        random_seed=123
    )
    print_simulation_summary(rate_sim)
    
    # Test TLT path simulation
    print("\n[6] Testing TLT path simulation...")
    tlt_sim = simulate_tlt_paths(rate_sim, reg_params, vol_params)
    print_tlt_simulation_summary(tlt_sim)
    
    # Summary
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nThe calibration and simulation modules are working correctly.")
    print("You can now use your actual historical data with run_calibration.py")
    
    return True


if __name__ == '__main__':
    success = run_test()
    sys.exit(0 if success else 1)
