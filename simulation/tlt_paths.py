"""
TLT Price Path Simulation

This module converts simulated yield paths to TLT price paths
using the calibrated regression model.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration.tlt_regression import TLTRegressionParams, predict_tlt_price
from calibration.volatility import VolatilityParams, estimate_iv_path
from simulation.rate_paths import SimulationResult


@dataclass
class TLTSimulationResult:
    """Container for TLT simulation results."""
    
    # Price paths
    tlt_paths: np.ndarray  # Shape: (n_steps + 1, n_paths)
    
    # IV paths (if computed)
    iv_paths: Optional[np.ndarray] = None
    
    # Underlying yield paths
    yield_paths_avg: Optional[np.ndarray] = None
    
    # Metadata
    n_steps: int = 0
    n_paths: int = 0
    dt: float = 0.0
    initial_price: float = 0.0
    
    @property
    def time_grid(self) -> np.ndarray:
        """Time points in years."""
        return np.arange(self.n_steps + 1) * self.dt
    
    @property
    def time_grid_days(self) -> np.ndarray:
        """Time points in trading days."""
        return np.arange(self.n_steps + 1)
    
    def terminal_prices(self) -> np.ndarray:
        """Terminal price distribution."""
        return self.tlt_paths[-1, :]
    
    def terminal_returns(self) -> np.ndarray:
        """Terminal return distribution."""
        return (self.tlt_paths[-1, :] - self.initial_price) / self.initial_price
    
    def price_at_step(self, step: int) -> np.ndarray:
        """Price distribution at a specific step."""
        return self.tlt_paths[step, :]
    
    def iv_at_step(self, step: int) -> Optional[np.ndarray]:
        """IV distribution at a specific step."""
        if self.iv_paths is not None:
            return self.iv_paths[step, :]
        return None
    
    def percentile_paths(
        self,
        percentiles: list = [5, 25, 50, 75, 95]
    ) -> dict:
        """Compute percentile price paths."""
        result = {}
        for p in percentiles:
            result[p] = np.percentile(self.tlt_paths, p, axis=1)
        return result
    
    def return_percentiles(
        self,
        percentiles: list = [5, 25, 50, 75, 95]
    ) -> dict:
        """Terminal return percentiles."""
        returns = self.terminal_returns()
        return {p: np.percentile(returns, p) for p in percentiles}


def convert_yields_to_tlt(
    yield_paths: np.ndarray,
    regression_params: TLTRegressionParams
) -> np.ndarray:
    """
    Convert yield paths to TLT price paths.
    
    Parameters
    ----------
    yield_paths : np.ndarray
        Average yield paths, shape (n_steps + 1, n_paths).
    regression_params : TLTRegressionParams
        Calibrated TLT regression model.
        
    Returns
    -------
    np.ndarray
        TLT price paths, same shape as input.
    """
    # Flatten for prediction, then reshape
    original_shape = yield_paths.shape
    yields_flat = yield_paths.flatten()
    
    prices_flat = predict_tlt_price(yields_flat, regression_params)
    
    return prices_flat.reshape(original_shape)


def simulate_tlt_paths(
    rate_simulation: SimulationResult,
    regression_params: TLTRegressionParams,
    vol_params: Optional[VolatilityParams] = None,
    anchor_price: Optional[float] = None
) -> TLTSimulationResult:
    """
    Generate TLT price paths from rate simulation.
    
    Parameters
    ----------
    rate_simulation : SimulationResult
        Output from yield simulation.
    regression_params : TLTRegressionParams
        TLT regression parameters.
    vol_params : VolatilityParams, optional
        Volatility parameters for IV estimation.
    anchor_price : float, optional
        If provided, scales the price paths so the initial price matches
        this observed value.
        
    Returns
    -------
    TLTSimulationResult
        TLT price and IV paths.
    """
    # Convert yields to prices
    tlt_paths = convert_yields_to_tlt(
        rate_simulation.paths_avg,
        regression_params
    )

    if anchor_price is not None:
        model_initial = tlt_paths[0, 0]
        if abs(model_initial) > 1e-10:
            scale = anchor_price / model_initial
            tlt_paths = tlt_paths * scale
    
    # Estimate IV paths if vol params provided
    iv_paths = None
    if vol_params is not None:
        iv_paths = np.zeros_like(tlt_paths)
        for path_idx in range(rate_simulation.n_paths):
            iv_paths[:, path_idx] = estimate_iv_path(
                tlt_paths[:, path_idx],
                vol_params
            )
    
    return TLTSimulationResult(
        tlt_paths=tlt_paths,
        iv_paths=iv_paths,
        yield_paths_avg=rate_simulation.paths_avg,
        n_steps=rate_simulation.n_steps,
        n_paths=rate_simulation.n_paths,
        dt=rate_simulation.dt,
        initial_price=tlt_paths[0, 0]
    )


def compute_tlt_statistics(result: TLTSimulationResult) -> dict:
    """
    Compute summary statistics for TLT simulation.
    
    Parameters
    ----------
    result : TLTSimulationResult
        Simulation output.
        
    Returns
    -------
    dict
        Summary statistics.
    """
    terminal_prices = result.terminal_prices()
    terminal_returns = result.terminal_returns()
    
    stats = {
        'initial_price': result.initial_price,
        'horizon_days': result.n_steps,
        'n_paths': result.n_paths,
        
        'terminal_price': {
            'mean': np.mean(terminal_prices),
            'std': np.std(terminal_prices),
            'min': np.min(terminal_prices),
            'max': np.max(terminal_prices),
            'p5': np.percentile(terminal_prices, 5),
            'p25': np.percentile(terminal_prices, 25),
            'p50': np.percentile(terminal_prices, 50),
            'p75': np.percentile(terminal_prices, 75),
            'p95': np.percentile(terminal_prices, 95)
        },
        
        'terminal_return': {
            'mean': np.mean(terminal_returns),
            'std': np.std(terminal_returns),
            'min': np.min(terminal_returns),
            'max': np.max(terminal_returns),
            'p5': np.percentile(terminal_returns, 5),
            'p25': np.percentile(terminal_returns, 25),
            'p50': np.percentile(terminal_returns, 50),
            'p75': np.percentile(terminal_returns, 75),
            'p95': np.percentile(terminal_returns, 95),
            'prob_positive': np.mean(terminal_returns > 0),
            'prob_negative': np.mean(terminal_returns < 0)
        }
    }
    
    # Add IV statistics if available
    if result.iv_paths is not None:
        terminal_iv = result.iv_at_step(-1)
        stats['terminal_iv'] = {
            'mean': np.mean(terminal_iv),
            'std': np.std(terminal_iv),
            'p5': np.percentile(terminal_iv, 5),
            'p95': np.percentile(terminal_iv, 95)
        }
    
    return stats


def print_tlt_simulation_summary(result: TLTSimulationResult) -> None:
    """Print TLT simulation summary."""
    stats = compute_tlt_statistics(result)
    
    print("=" * 60)
    print("TLT PRICE SIMULATION SUMMARY")
    print("=" * 60)
    
    print(f"\nSimulation Parameters:")
    print(f"  Initial Price:     ${result.initial_price:.2f}")
    print(f"  Horizon:           {result.n_steps} trading days")
    print(f"  Paths:             {result.n_paths:,}")
    
    print(f"\nTerminal Price Distribution:")
    tp = stats['terminal_price']
    print(f"  Mean:              ${tp['mean']:.2f}")
    print(f"  Std Dev:           ${tp['std']:.2f}")
    print(f"  5th percentile:    ${tp['p5']:.2f}")
    print(f"  Median:            ${tp['p50']:.2f}")
    print(f"  95th percentile:   ${tp['p95']:.2f}")
    print(f"  Range:             ${tp['min']:.2f} - ${tp['max']:.2f}")
    
    print(f"\nTerminal Return Distribution:")
    tr = stats['terminal_return']
    print(f"  Mean:              {tr['mean']*100:+.2f}%")
    print(f"  Std Dev:           {tr['std']*100:.2f}%")
    print(f"  5th percentile:    {tr['p5']*100:+.2f}%")
    print(f"  Median:            {tr['p50']*100:+.2f}%")
    print(f"  95th percentile:   {tr['p95']*100:+.2f}%")
    print(f"  P(positive):       {tr['prob_positive']*100:.1f}%")
    print(f"  P(negative):       {tr['prob_negative']*100:.1f}%")
    
    if 'terminal_iv' in stats:
        print(f"\nTerminal IV Distribution:")
        iv = stats['terminal_iv']
        print(f"  Mean:              {iv['mean']*100:.1f}%")
        print(f"  5th-95th range:    {iv['p5']*100:.1f}% - {iv['p95']*100:.1f}%")
    
    print("=" * 60)


def extract_scenario_paths(
    result: TLTSimulationResult,
    n_scenarios: int = 5,
    method: str = 'percentile'
) -> dict:
    """
    Extract representative scenario paths for visualization.
    
    Parameters
    ----------
    result : TLTSimulationResult
        Simulation output.
    n_scenarios : int
        Number of scenarios to extract.
    method : str
        Selection method:
        - 'percentile': Select paths closest to key percentiles
        - 'random': Random selection
        - 'extreme': Include extreme paths
        
    Returns
    -------
    dict
        Dictionary with scenario paths.
    """
    terminal_returns = result.terminal_returns()
    
    if method == 'percentile':
        # Select paths at specific percentiles
        percentiles = np.linspace(10, 90, n_scenarios)
        scenarios = {}
        
        for i, p in enumerate(percentiles):
            target_return = np.percentile(terminal_returns, p)
            # Find path closest to this percentile
            idx = np.argmin(np.abs(terminal_returns - target_return))
            scenarios[f'p{int(p)}'] = {
                'path_idx': idx,
                'tlt_path': result.tlt_paths[:, idx],
                'yield_path': result.yield_paths_avg[:, idx] if result.yield_paths_avg is not None else None,
                'terminal_return': terminal_returns[idx]
            }
            
    elif method == 'extreme':
        # Include median plus extreme scenarios
        scenarios = {}
        
        # Worst case
        idx_worst = np.argmin(terminal_returns)
        scenarios['worst'] = {
            'path_idx': idx_worst,
            'tlt_path': result.tlt_paths[:, idx_worst],
            'terminal_return': terminal_returns[idx_worst]
        }
        
        # Best case
        idx_best = np.argmax(terminal_returns)
        scenarios['best'] = {
            'path_idx': idx_best,
            'tlt_path': result.tlt_paths[:, idx_best],
            'terminal_return': terminal_returns[idx_best]
        }
        
        # Median
        idx_median = np.argmin(np.abs(terminal_returns - np.median(terminal_returns)))
        scenarios['median'] = {
            'path_idx': idx_median,
            'tlt_path': result.tlt_paths[:, idx_median],
            'terminal_return': terminal_returns[idx_median]
        }
        
        # 5th and 95th percentile
        for p, name in [(5, 'p5'), (95, 'p95')]:
            target = np.percentile(terminal_returns, p)
            idx = np.argmin(np.abs(terminal_returns - target))
            scenarios[name] = {
                'path_idx': idx,
                'tlt_path': result.tlt_paths[:, idx],
                'terminal_return': terminal_returns[idx]
            }
            
    else:  # random
        indices = np.random.choice(result.n_paths, n_scenarios, replace=False)
        scenarios = {
            f'random_{i}': {
                'path_idx': idx,
                'tlt_path': result.tlt_paths[:, idx],
                'terminal_return': terminal_returns[idx]
            }
            for i, idx in enumerate(indices)
        }
    
    return scenarios
