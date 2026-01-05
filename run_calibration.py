"""
Main Calibration Script

This script runs the complete Phase 1 calibration pipeline:
1. Load and validate historical data
2. Calibrate bivariate Vasicek model
3. Calibrate TLT regression
4. Calibrate volatility/IV model
5. Save calibrated parameters

Usage:
    python run_calibration.py --data path/to/data.csv [--output params.json]
"""

import argparse
import json
import sys
from pathlib import Path

# Add script directory to path BEFORE other imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import date
import numpy as np
import pandas as pd

from config import ModelConfig, CalibrationConfig, DataConfig
from data.loader import (
    load_single_file,
    filter_calibration_period,
    compute_derived_features,
    validate_data,
    print_validation_report
)
from calibration.vasicek import (
    calibrate_bivariate_vasicek,
    print_calibration_report as print_vasicek_report
)
from calibration.tlt_regression import (
    calibrate_tlt_regression,
    print_regression_report
)
from calibration.volatility import (
    calibrate_iv_price_relationship,
    print_volatility_report
)


def run_full_calibration(
    data_path: str,
    config: ModelConfig = None,
    verbose: bool = True
) -> dict:
    """
    Run the complete calibration pipeline.
    
    Parameters
    ----------
    data_path : str
        Path to CSV data file.
    config : ModelConfig, optional
        Configuration object. Uses defaults if not provided.
    verbose : bool
        Whether to print reports.
        
    Returns
    -------
    dict
        Dictionary containing all calibrated parameters.
    """
    if config is None:
        config = ModelConfig()
    
    results = {}
    
    # =========================================================================
    # Step 1: Load and Validate Data
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 1: LOADING AND VALIDATING DATA")
        print("=" * 70)
    
    # Try to load data
    try:
        df = load_single_file(data_path, config.data)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nPlease ensure your CSV has the following columns:")
        print(f"  - {config.data.date_column}")
        print(f"  - {config.data.yield_20y_column}")
        print(f"  - {config.data.yield_30y_column}")
        print(f"  - {config.data.tlt_close_column}")
        return None
    
    if verbose:
        print(f"Loaded {len(df)} observations from {data_path}")
        print(f"  Date range in file: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Calibration window: {config.calibration.calibration_start} to {config.calibration.calibration_end}")
        print(f"  COVID exclusion:    {config.calibration.covid_start} to {config.calibration.covid_end}")
    
    # Filter to calibration period
    df_calib = filter_calibration_period(df, config.calibration, exclude_covid=True)
    
    if len(df_calib) < 50:
        print(f"\n⚠️  WARNING: Only {len(df_calib)} observations after filtering!")
        print("    This may be insufficient for reliable calibration.")
        print("    Consider adjusting calibration dates in config.py or via command line.")
        
        if len(df_calib) == 0:
            print("\n❌ ERROR: No data remains after filtering. Check date ranges.")
            print(f"    Your data: {df.index.min().date()} to {df.index.max().date()}")
            print(f"    Calibration window: {config.calibration.calibration_start} to {config.calibration.calibration_end}")
            return None
    
    # Add derived features
    df_calib = compute_derived_features(df_calib)
    df_calib = df_calib.dropna()
    
    if verbose:
        print(f"\nAfter processing: {len(df_calib)} observations available for calibration")
        validation = validate_data(df_calib)
        print_validation_report(validation)
    
    results['data_info'] = {
        'source_file': data_path,
        'total_observations': len(df),
        'calibration_observations': len(df_calib),
        'date_range': [
            df_calib.index.min().strftime('%Y-%m-%d'),
            df_calib.index.max().strftime('%Y-%m-%d')
        ],
        'covid_excluded': [
            config.calibration.covid_start.strftime('%Y-%m-%d'),
            config.calibration.covid_end.strftime('%Y-%m-%d')
        ]
    }
    
    # =========================================================================
    # Step 2: Calibrate Bivariate Vasicek Model
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: CALIBRATING BIVARIATE VASICEK MODEL")
        print("=" * 70)
    
    # Extract yield series (convert from percentage if needed)
    yields_20y = df_calib['yield_20y'].values
    yields_30y = df_calib['yield_30y'].values
    
    # Check if yields are in percentage form
    if yields_20y.mean() > 0.5:  # Likely in percentage form
        if verbose:
            print("Note: Converting yields from percentage to decimal form")
        yields_20y = yields_20y / 100
        yields_30y = yields_30y / 100
    
    dt = 1.0 / config.simulation.trading_days_per_year
    
    vasicek_params, vasicek_diag = calibrate_bivariate_vasicek(
        yields_20y, yields_30y, dt, use_mle=config.calibration.use_mle
    )
    
    if verbose:
        print_vasicek_report(vasicek_params, vasicek_diag)
    
    results['vasicek_params'] = vasicek_params.to_dict()
    results['vasicek_diagnostics'] = {
        k: v for k, v in vasicek_diag.items() 
        if not isinstance(v, np.ndarray)
    }
    
    # =========================================================================
    # Step 3: Calibrate TLT Regression
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: CALIBRATING TLT REGRESSION MODEL")
        print("=" * 70)
    
    yields_avg = (yields_20y + yields_30y) / 2
    tlt_prices = df_calib['tlt_close'].values
    
    regression_params, regression_diag = calibrate_tlt_regression(
        yields_avg, tlt_prices, include_convexity=True
    )
    
    if verbose:
        print_regression_report(regression_params, regression_diag)
    
    results['regression_params'] = regression_params.to_dict()
    results['regression_diagnostics'] = {
        k: v for k, v in regression_diag.items()
        if not isinstance(v, np.ndarray)
    }
    
    # =========================================================================
    # Step 4: Calibrate Volatility Model
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 4: CALIBRATING VOLATILITY MODEL")
        print("=" * 70)
    
    vol_params, vol_diag = calibrate_iv_price_relationship(
        tlt_prices,
        window=config.calibration.vol_window_days,
        annualization=config.calibration.vol_annualization_factor
    )
    
    if verbose:
        print_volatility_report(vol_params, vol_diag)
    
    results['volatility_params'] = vol_params.to_dict()
    results['volatility_diagnostics'] = {
        k: v for k, v in vol_diag.items()
        if not isinstance(v, (np.ndarray, dict))
    }
    
    # =========================================================================
    # Step 5: Store Current Market Values
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("CURRENT MARKET VALUES (Last Observation)")
        print("=" * 70)
    
    current_values = {
        'date': df_calib.index[-1].strftime('%Y-%m-%d'),
        'yield_20y': float(yields_20y[-1]),
        'yield_30y': float(yields_30y[-1]),
        'yield_avg': float(yields_avg[-1]),
        'tlt_price': float(tlt_prices[-1])
    }
    
    if verbose:
        print(f"  Date:              {current_values['date']}")
        print(f"  20Y Yield:         {current_values['yield_20y']*100:.3f}%")
        print(f"  30Y Yield:         {current_values['yield_30y']*100:.3f}%")
        print(f"  Average Yield:     {current_values['yield_avg']*100:.3f}%")
        print(f"  TLT Price:         ${current_values['tlt_price']:.2f}")
    
    results['current_values'] = current_values
    
    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("CALIBRATION COMPLETE")
        print("=" * 70)
        print("\nCalibrated components:")
        print("  ✓ Bivariate Vasicek model for 20Y/30Y yields")
        print("  ✓ TLT regression on average yield")
        print("  ✓ IV-price relationship")
        print(f"\nReady for Phase 2: Simulation")
    
    return results


def save_calibration(results: dict, output_path: str) -> None:
    """Save calibration results to JSON file."""
    
    def make_serializable(obj):
        """Recursively convert objects to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return make_serializable(obj.__dict__)
        else:
            return obj
    
    serializable_results = make_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nCalibration saved to: {output_path}")


def load_calibration(input_path: str) -> dict:
    """Load calibration results from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Run TLT Options Model Calibration'
    )
    parser.add_argument(
        '--data', '-d',
        required=True,
        help='Path to CSV data file'
    )
    parser.add_argument(
        '--output', '-o',
        default='calibration_results.json',
        help='Output path for calibration results'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--date-col',
        default='date',
        help='Name of date column in CSV'
    )
    parser.add_argument(
        '--yield-20y-col',
        default='yield_20y',
        help='Name of 20Y yield column'
    )
    parser.add_argument(
        '--yield-30y-col',
        default='yield_30y',
        help='Name of 30Y yield column'
    )
    parser.add_argument(
        '--tlt-col',
        default='tlt_close',
        help='Name of TLT price column'
    )
    parser.add_argument(
        '--start-date',
        default=None,
        help='Calibration start date (YYYY-MM-DD). Default: 2019-01-01'
    )
    parser.add_argument(
        '--end-date',
        default=None,
        help='Calibration end date (YYYY-MM-DD). Default: 2024-12-31'
    )
    parser.add_argument(
        '--no-covid-exclusion',
        action='store_true',
        help='Do not exclude COVID period from calibration'
    )
    
    args = parser.parse_args()
    
    # Build config with column mappings
    data_config = DataConfig(
        date_column=args.date_col,
        yield_20y_column=args.yield_20y_col,
        yield_30y_column=args.yield_30y_col,
        tlt_close_column=args.tlt_col
    )
    
    # Build calibration config with date overrides
    calib_kwargs = {}
    if args.start_date:
        from datetime import datetime
        calib_kwargs['calibration_start'] = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    if args.end_date:
        from datetime import datetime
        calib_kwargs['calibration_end'] = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    if args.no_covid_exclusion:
        # Set COVID period to a range that won't exclude anything
        calib_kwargs['covid_start'] = date(1900, 1, 1)
        calib_kwargs['covid_end'] = date(1900, 1, 2)
    
    calib_config = CalibrationConfig(**calib_kwargs) if calib_kwargs else CalibrationConfig()
    
    config = ModelConfig(data=data_config, calibration=calib_config)
    
    # Run calibration
    results = run_full_calibration(
        args.data,
        config=config,
        verbose=not args.quiet
    )
    
    if results is not None:
        save_calibration(results, args.output)
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
