"""
Data Loading and Preprocessing

This module handles loading historical yield and TLT data,
cleaning, date filtering, and COVID period exclusion.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
from typing import Optional, Tuple, Union
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DataConfig, CalibrationConfig, DEFAULT_CONFIG


def load_yield_data(
    filepath: Union[str, Path],
    config: DataConfig = DEFAULT_CONFIG.data
) -> pd.DataFrame:
    """
    Load yield data from CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the yield data CSV file.
    config : DataConfig
        Configuration with column name mappings.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and yield columns.
    """
    df = pd.read_csv(filepath)
    
    # Parse dates
    df[config.date_column] = pd.to_datetime(
        df[config.date_column], 
        format=config.date_format
    )
    df = df.set_index(config.date_column)
    df.index.name = 'date'
    
    # Validate required columns exist
    required_cols = [config.yield_20y_column, config.yield_30y_column]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Rename to standard names
    df = df.rename(columns={
        config.yield_20y_column: 'yield_20y',
        config.yield_30y_column: 'yield_30y'
    })
    
    # Keep only yield columns
    df = df[['yield_20y', 'yield_30y']].copy()
    
    # Sort by date
    df = df.sort_index()
    
    return df


def load_tlt_data(
    filepath: Union[str, Path],
    config: DataConfig = DEFAULT_CONFIG.data
) -> pd.DataFrame:
    """
    Load TLT price data from CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the TLT data CSV file.
    config : DataConfig
        Configuration with column name mappings.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and TLT close price.
    """
    df = pd.read_csv(filepath)
    
    # Parse dates
    df[config.date_column] = pd.to_datetime(
        df[config.date_column],
        format=config.date_format
    )
    df = df.set_index(config.date_column)
    df.index.name = 'date'
    
    # Validate required column
    if config.tlt_close_column not in df.columns:
        raise ValueError(f"Missing required column: {config.tlt_close_column}")
    
    # Rename to standard name
    df = df.rename(columns={config.tlt_close_column: 'tlt_close'})
    
    # Keep only TLT column
    df = df[['tlt_close']].copy()
    
    # Sort by date
    df = df.sort_index()
    
    return df


def load_combined_data(
    yield_filepath: Union[str, Path],
    tlt_filepath: Union[str, Path],
    config: DataConfig = DEFAULT_CONFIG.data
) -> pd.DataFrame:
    """
    Load and merge yield and TLT data.
    
    Parameters
    ----------
    yield_filepath : str or Path
        Path to the yield data CSV file.
    tlt_filepath : str or Path
        Path to the TLT data CSV file.
    config : DataConfig
        Configuration with column name mappings.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with yields and TLT price, aligned by date.
    """
    yields = load_yield_data(yield_filepath, config)
    tlt = load_tlt_data(tlt_filepath, config)
    
    # Inner join on dates
    df = yields.join(tlt, how='inner')
    
    # Report any date mismatches
    yield_dates = set(yields.index)
    tlt_dates = set(tlt.index)
    
    only_yields = yield_dates - tlt_dates
    only_tlt = tlt_dates - yield_dates
    
    if only_yields:
        warnings.warn(f"{len(only_yields)} dates in yield data but not TLT data")
    if only_tlt:
        warnings.warn(f"{len(only_tlt)} dates in TLT data but not yield data")
    
    return df


def load_single_file(
    filepath: Union[str, Path],
    config: DataConfig = DEFAULT_CONFIG.data
) -> pd.DataFrame:
    """
    Load data from a single CSV file containing all columns.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the combined data CSV file.
    config : DataConfig
        Configuration with column name mappings.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with yields and TLT price.
    """
    df = pd.read_csv(filepath)
    
    # Parse dates - try specified format first, then infer
    try:
        df[config.date_column] = pd.to_datetime(
            df[config.date_column],
            format=config.date_format
        )
    except (ValueError, TypeError):
        # Let pandas infer the format
        df[config.date_column] = pd.to_datetime(df[config.date_column])
    
    df = df.set_index(config.date_column)
    df.index.name = 'date'
    
    # Validate required columns (case-insensitive search)
    available_cols = {c.lower(): c for c in df.columns}
    
    def find_column(requested: str) -> str:
        """Find column, trying exact match first then case-insensitive."""
        if requested in df.columns:
            return requested
        if requested.lower() in available_cols:
            return available_cols[requested.lower()]
        raise ValueError(f"Column '{requested}' not found. Available: {list(df.columns)}")
    
    yield_20y_col = find_column(config.yield_20y_column)
    yield_30y_col = find_column(config.yield_30y_column)
    tlt_close_col = find_column(config.tlt_close_column)
    
    # Rename to standard names
    df = df.rename(columns={
        yield_20y_col: 'yield_20y',
        yield_30y_col: 'yield_30y',
        tlt_close_col: 'tlt_close'
    })
    
    # Keep only required columns
    df = df[['yield_20y', 'yield_30y', 'tlt_close']].copy()
    
    # Convert to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with missing data
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        dropped = initial_len - len(df)
        print(f"  Note: Dropped {dropped} rows with missing/invalid data")
    
    # Sort by date
    df = df.sort_index()
    
    return df


def filter_calibration_period(
    df: pd.DataFrame,
    config: CalibrationConfig = DEFAULT_CONFIG.calibration,
    exclude_covid: bool = True
) -> pd.DataFrame:
    """
    Filter data to calibration period, optionally excluding COVID era.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full historical data with DatetimeIndex.
    config : CalibrationConfig
        Calibration configuration with date ranges.
    exclude_covid : bool
        Whether to exclude the COVID period.
        
    Returns
    -------
    pd.DataFrame
        Filtered data for calibration.
    """
    # Convert dates to timestamps for comparison
    start = pd.Timestamp(config.calibration_start)
    end = pd.Timestamp(config.calibration_end)
    
    # Filter to calibration window
    mask = (df.index >= start) & (df.index <= end)
    
    if exclude_covid:
        covid_start = pd.Timestamp(config.covid_start)
        covid_end = pd.Timestamp(config.covid_end)
        covid_mask = (df.index >= covid_start) & (df.index <= covid_end)
        mask = mask & ~covid_mask
    
    filtered = df[mask].copy()
    
    print(f"Calibration data: {len(filtered)} observations")
    print(f"  Date range: {filtered.index.min().date()} to {filtered.index.max().date()}")
    if exclude_covid:
        print(f"  Excluded COVID period: {config.covid_start} to {config.covid_end}")
    
    return filtered


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features needed for calibration.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with yield_20y, yield_30y, tlt_close columns.
        
    Returns
    -------
    pd.DataFrame
        Data with additional derived columns.
    """
    df = df.copy()
    
    # Average yield (used in TLT regression)
    df['yield_avg'] = (df['yield_20y'] + df['yield_30y']) / 2
    
    # Yield changes
    df['yield_20y_change'] = df['yield_20y'].diff()
    df['yield_30y_change'] = df['yield_30y'].diff()
    df['yield_avg_change'] = df['yield_avg'].diff()
    
    # TLT returns
    df['tlt_return'] = df['tlt_close'].pct_change()
    df['tlt_log_return'] = np.log(df['tlt_close']).diff()
    
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate data quality and return diagnostics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to validate.
        
    Returns
    -------
    dict
        Validation results and statistics.
    """
    results = {
        'n_rows': len(df),
        'date_range': (df.index.min(), df.index.max()),
        'missing_values': df.isnull().sum().to_dict(),
        'statistics': df.describe().to_dict()
    }
    
    # Check for gaps in dates (more than 5 trading days)
    date_diffs = df.index.to_series().diff().dt.days
    large_gaps = date_diffs[date_diffs > 5]
    results['date_gaps'] = [
        (idx.date(), int(gap)) 
        for idx, gap in large_gaps.items()
    ]
    
    # Check for outliers (beyond 5 std from mean)
    for col in ['yield_20y', 'yield_30y', 'tlt_close']:
        if col in df.columns:
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            outliers = df.index[z_scores.abs() > 5]
            results[f'{col}_outliers'] = [d.date() for d in outliers]
    
    return results


def print_validation_report(validation: dict) -> None:
    """Print a human-readable validation report."""
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\nObservations: {validation['n_rows']}")
    print(f"Date range: {validation['date_range'][0].date()} to {validation['date_range'][1].date()}")
    
    print("\nMissing values:")
    for col, count in validation['missing_values'].items():
        status = "✓" if count == 0 else f"⚠ {count} missing"
        print(f"  {col}: {status}")
    
    if validation['date_gaps']:
        print(f"\nDate gaps (>5 days): {len(validation['date_gaps'])}")
        for date, gap in validation['date_gaps'][:5]:
            print(f"  {date}: {gap} days")
        if len(validation['date_gaps']) > 5:
            print(f"  ... and {len(validation['date_gaps']) - 5} more")
    else:
        print("\nDate gaps: None detected")
    
    print("\nOutlier check (>5 std):")
    for key in ['yield_20y_outliers', 'yield_30y_outliers', 'tlt_close_outliers']:
        if key in validation:
            outliers = validation[key]
            col = key.replace('_outliers', '')
            if outliers:
                print(f"  {col}: {len(outliers)} outliers")
            else:
                print(f"  {col}: ✓ No outliers")
    
    print("=" * 60)


# Convenience function for typical workflow
def prepare_calibration_data(
    filepath: Union[str, Path],
    config: Optional[CalibrationConfig] = None,
    data_config: Optional[DataConfig] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Complete data preparation pipeline for calibration.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the data CSV file.
    config : CalibrationConfig, optional
        Calibration configuration.
    data_config : DataConfig, optional
        Data loading configuration.
    validate : bool
        Whether to run and print validation.
        
    Returns
    -------
    pd.DataFrame
        Cleaned, filtered, and enriched data ready for calibration.
    """
    if config is None:
        config = DEFAULT_CONFIG.calibration
    if data_config is None:
        data_config = DEFAULT_CONFIG.data
    
    # Load data
    df = load_single_file(filepath, data_config)
    
    # Filter to calibration period
    df = filter_calibration_period(df, config, exclude_covid=True)
    
    # Add derived features
    df = compute_derived_features(df)
    
    # Drop rows with NaN from differencing
    df = df.dropna()
    
    if validate:
        validation = validate_data(df)
        print_validation_report(validation)
    
    return df
