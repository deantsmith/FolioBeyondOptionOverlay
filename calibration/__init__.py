"""Calibration modules for interest rate and TLT models."""

from .vasicek import (
    VasicekParams,
    BivariateVasicekParams,
    calibrate_univariate_vasicek_ols,
    calibrate_univariate_vasicek_mle,
    calibrate_bivariate_vasicek,
    print_calibration_report
)

from .tlt_regression import (
    TLTRegressionParams,
    calibrate_tlt_regression,
    calibrate_return_regression,
    predict_tlt_price,
    predict_tlt_return,
    compute_duration_convexity,
    print_regression_report
)

from .volatility import (
    VolatilityParams,
    compute_realized_volatility,
    compute_volatility_statistics,
    calibrate_iv_price_relationship,
    estimate_iv,
    estimate_iv_path,
    print_volatility_report
)

__all__ = [
    # Vasicek
    'VasicekParams',
    'BivariateVasicekParams',
    'calibrate_univariate_vasicek_ols',
    'calibrate_univariate_vasicek_mle',
    'calibrate_bivariate_vasicek',
    'print_calibration_report',
    
    # TLT Regression
    'TLTRegressionParams',
    'calibrate_tlt_regression',
    'calibrate_return_regression',
    'predict_tlt_price',
    'predict_tlt_return',
    'compute_duration_convexity',
    'print_regression_report',
    
    # Volatility
    'VolatilityParams',
    'compute_realized_volatility',
    'compute_volatility_statistics',
    'calibrate_iv_price_relationship',
    'estimate_iv',
    'estimate_iv_path',
    'print_volatility_report'
]
