"""
Configuration for TLT Options Overlay Model

This module contains all parameters, constants, and configuration settings
for the options overlay strategy model.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class CalibrationConfig:
    """Parameters for model calibration."""
    
    # Calibration date range
    calibration_start: date = date(2019, 1, 1)
    calibration_end: date = date(2024, 12, 31)
    
    # COVID exclusion period
    covid_start: date = date(2020, 3, 1)
    covid_end: date = date(2021, 12, 31)
    
    # Vasicek calibration
    use_mle: bool = True  # Use MLE; if False, use OLS approximation
    
    # Realized volatility estimation
    vol_window_days: int = 21  # Trading days for rolling vol estimate
    vol_annualization_factor: float = 252.0  # Trading days per year


@dataclass
class SimulationConfig:
    """Parameters for Monte Carlo simulation."""
    
    # Simulation settings
    n_paths: int = 10_000
    random_seed: Optional[int] = 42  # Set to None for non-reproducible
    
    # Time discretization
    dt_days: int = 1  # Daily steps
    trading_days_per_year: float = 252.0
    
    @property
    def dt(self) -> float:
        """Time step in years."""
        return self.dt_days / self.trading_days_per_year


@dataclass
class OptionConfig:
    """Parameters for option modeling."""
    
    # Typical option horizons (DTE)
    min_dte: int = 30
    max_dte: int = 90
    
    # Risk-free rate for Black-Scholes (annualized)
    # Will be updated with current short rate
    risk_free_rate: float = 0.05
    
    # IV estimation
    iv_base_percentile: float = 50.0  # Median realized vol as base
    iv_price_sensitivity: float = -0.02  # IV change per 1% TLT decline


@dataclass
class ExitRuleConfig:
    """Parameters for position exit rules."""
    
    # Profit target: close when spread value <= X% of credit received
    profit_target_pct: float = 0.50  # Close at 50% profit
    
    # Loss limit: close when spread value >= X * credit received
    loss_limit_multiple: float = 2.0  # Close at 200% of credit (100% loss)
    
    # DTE threshold: close with N days remaining
    dte_close_threshold: int = 7
    
    # Enable/disable individual rules
    use_profit_target: bool = True
    use_loss_limit: bool = True
    use_dte_threshold: bool = True


@dataclass
class RiskConfig:
    """Risk budget and constraint parameters."""
    
    # Base portfolio
    portfolio_yield_annual: float = 0.0475  # 4.75%
    
    # Loss budget as fraction of yield
    loss_budget_fraction: float = 0.50  # 50% of yield
    
    # CVaR confidence level
    cvar_confidence: float = 0.95
    
    # Minimum probability of profit
    min_pop: float = 0.70  # 70%
    
    @property
    def annual_loss_budget(self) -> float:
        """Annual loss budget as fraction of NAV."""
        return self.portfolio_yield_annual * self.loss_budget_fraction
    
    @property
    def monthly_cvar_budget(self) -> float:
        """
        Monthly CVaR budget as fraction of NAV.
        Slightly higher than annual/12 to account for clustering.
        """
        return self.annual_loss_budget / 12 * 1.2


@dataclass
class StrategyConfig:
    """Parameters for strategy definition and evaluation."""
    
    # Contract sizing
    contracts_per_trade: int = 10
    tlt_multiplier: int = 100  # Shares per contract
    
    # Spread parameters to evaluate
    spread_widths: List[int] = field(default_factory=lambda: [2, 3, 5])
    
    # Delta targets for short strikes (absolute value)
    put_delta_targets: List[float] = field(default_factory=lambda: [0.15, 0.20, 0.25, 0.30])
    call_delta_targets: List[float] = field(default_factory=lambda: [0.15, 0.20, 0.25, 0.30])


@dataclass
class DataConfig:
    """Data file paths and column mappings."""
    
    # File paths (to be updated with actual paths)
    yield_data_path: str = "data/yields.csv"
    tlt_data_path: str = "data/tlt.csv"
    
    # Column names (adjust to match your data)
    date_column: str = "date"
    yield_20y_column: str = "yield_20y"
    yield_30y_column: str = "yield_30y"
    tlt_close_column: str = "tlt_close"
    
    # Date format in CSV
    date_format: str = "%Y-%m-%d"


@dataclass
class ModelConfig:
    """Master configuration combining all sub-configs."""
    
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    option: OptionConfig = field(default_factory=OptionConfig)
    exit_rules: ExitRuleConfig = field(default_factory=ExitRuleConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    data: DataConfig = field(default_factory=DataConfig)


# Default configuration instance
DEFAULT_CONFIG = ModelConfig()


def create_config(**kwargs) -> ModelConfig:
    """
    Create a configuration with optional overrides.
    
    Example:
        config = create_config(
            simulation=SimulationConfig(n_paths=50000),
            risk=RiskConfig(portfolio_yield_annual=0.05)
        )
    """
    return ModelConfig(**kwargs)
