"""
Interest Rate Path Simulation

This module generates Monte Carlo simulations of correlated
20-year and 30-year Treasury yield paths using the calibrated
bivariate Vasicek model.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration.vasicek import BivariateVasicekParams, vasicek_conditional_moments


def simulate_univariate_vasicek(
    kappa: float,
    theta: float,
    sigma: float,
    r0: float,
    n_steps: int,
    dt: float,
    n_paths: int,
    random_state: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Simulate paths from univariate Vasicek model.
    
    Uses exact discretization (not Euler) for accuracy.
    
    Parameters
    ----------
    kappa : float
        Mean reversion speed.
    theta : float
        Long-run mean.
    sigma : float
        Volatility.
    r0 : float
        Initial rate.
    n_steps : int
        Number of time steps.
    dt : float
        Time step in years.
    n_paths : int
        Number of simulation paths.
    random_state : np.random.RandomState, optional
        Random state for reproducibility.
        
    Returns
    -------
    np.ndarray
        Simulated paths, shape (n_steps + 1, n_paths).
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    # Pre-compute constants
    exp_kappa_dt = np.exp(-kappa * dt)
    
    if kappa > 1e-10:
        conditional_var = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt))
    else:
        conditional_var = sigma**2 * dt
    
    conditional_std = np.sqrt(conditional_var)
    
    # Initialize paths
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0, :] = r0
    
    # Generate all random numbers upfront
    z = random_state.standard_normal((n_steps, n_paths))
    
    # Simulate
    for t in range(n_steps):
        # Conditional mean
        mean = paths[t, :] * exp_kappa_dt + theta * (1 - exp_kappa_dt)
        # Next step
        paths[t + 1, :] = mean + conditional_std * z[t, :]
    
    return paths


def simulate_bivariate_vasicek(
    params: BivariateVasicekParams,
    r0_20: float,
    r0_30: float,
    n_steps: int,
    dt: float,
    n_paths: int,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate correlated paths for 20Y and 30Y yields.
    
    Uses Cholesky decomposition to generate correlated innovations.
    
    Parameters
    ----------
    params : BivariateVasicekParams
        Calibrated model parameters.
    r0_20 : float
        Initial 20Y yield.
    r0_30 : float
        Initial 30Y yield.
    n_steps : int
        Number of time steps.
    dt : float
        Time step in years.
    n_paths : int
        Number of simulation paths.
    random_seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        20Y and 30Y yield paths, each shape (n_steps + 1, n_paths).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    p20 = params.params_20y
    p30 = params.params_30y
    rho = params.correlation
    
    # Pre-compute constants for each series
    exp_kappa_20 = np.exp(-p20.kappa * dt)
    exp_kappa_30 = np.exp(-p30.kappa * dt)
    
    if p20.kappa > 1e-10:
        var_20 = (p20.sigma**2 / (2 * p20.kappa)) * (1 - np.exp(-2 * p20.kappa * dt))
    else:
        var_20 = p20.sigma**2 * dt
        
    if p30.kappa > 1e-10:
        var_30 = (p30.sigma**2 / (2 * p30.kappa)) * (1 - np.exp(-2 * p30.kappa * dt))
    else:
        var_30 = p30.sigma**2 * dt
    
    std_20 = np.sqrt(var_20)
    std_30 = np.sqrt(var_30)
    
    # Cholesky decomposition for correlation
    # [z1]   [1      0        ] [eps1]
    # [z2] = [rho  sqrt(1-rho²)] [eps2]
    chol_21 = rho
    chol_22 = np.sqrt(1 - rho**2)
    
    # Initialize paths
    paths_20 = np.zeros((n_steps + 1, n_paths))
    paths_30 = np.zeros((n_steps + 1, n_paths))
    paths_20[0, :] = r0_20
    paths_30[0, :] = r0_30
    
    # Generate independent standard normals
    eps1 = np.random.standard_normal((n_steps, n_paths))
    eps2 = np.random.standard_normal((n_steps, n_paths))
    
    # Correlated innovations
    z1 = eps1
    z2 = chol_21 * eps1 + chol_22 * eps2
    
    # Simulate
    for t in range(n_steps):
        # 20Y
        mean_20 = paths_20[t, :] * exp_kappa_20 + p20.theta * (1 - exp_kappa_20)
        paths_20[t + 1, :] = mean_20 + std_20 * z1[t, :]
        
        # 30Y
        mean_30 = paths_30[t, :] * exp_kappa_30 + p30.theta * (1 - exp_kappa_30)
        paths_30[t + 1, :] = mean_30 + std_30 * z2[t, :]
    
    return paths_20, paths_30


def simulate_average_yield_paths(
    params: BivariateVasicekParams,
    r0_20: float,
    r0_30: float,
    n_steps: int,
    dt: float,
    n_paths: int,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate yield paths and compute average yield.
    
    Parameters
    ----------
    params : BivariateVasicekParams
        Model parameters.
    r0_20, r0_30 : float
        Initial yields.
    n_steps : int
        Number of time steps.
    dt : float
        Time step in years.
    n_paths : int
        Number of paths.
    random_seed : int, optional
        Random seed.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        20Y paths, 30Y paths, and average yield paths.
    """
    paths_20, paths_30 = simulate_bivariate_vasicek(
        params, r0_20, r0_30, n_steps, dt, n_paths, random_seed
    )
    
    paths_avg = (paths_20 + paths_30) / 2
    
    return paths_20, paths_30, paths_avg


@dataclass
class SimulationResult:
    """Container for simulation results."""
    
    paths_20y: np.ndarray
    paths_30y: np.ndarray
    paths_avg: np.ndarray
    
    n_steps: int
    n_paths: int
    dt: float
    
    initial_20y: float
    initial_30y: float
    
    @property
    def time_grid(self) -> np.ndarray:
        """Time points in years."""
        return np.arange(self.n_steps + 1) * self.dt
    
    @property
    def time_grid_days(self) -> np.ndarray:
        """Time points in trading days."""
        return np.arange(self.n_steps + 1)
    
    def terminal_distribution(self, series: str = 'avg') -> np.ndarray:
        """Get terminal yield distribution."""
        if series == '20y':
            return self.paths_20y[-1, :]
        elif series == '30y':
            return self.paths_30y[-1, :]
        else:
            return self.paths_avg[-1, :]
    
    def yield_change_distribution(self, series: str = 'avg') -> np.ndarray:
        """Get distribution of yield changes."""
        if series == '20y':
            return self.paths_20y[-1, :] - self.initial_20y
        elif series == '30y':
            return self.paths_30y[-1, :] - self.initial_30y
        else:
            initial_avg = (self.initial_20y + self.initial_30y) / 2
            return self.paths_avg[-1, :] - initial_avg
    
    def percentile_paths(
        self, 
        percentiles: list = [5, 25, 50, 75, 95],
        series: str = 'avg'
    ) -> dict:
        """Compute percentile paths for visualization."""
        if series == '20y':
            paths = self.paths_20y
        elif series == '30y':
            paths = self.paths_30y
        else:
            paths = self.paths_avg
        
        result = {}
        for p in percentiles:
            result[p] = np.percentile(paths, p, axis=1)
        return result


def run_simulation(
    params: BivariateVasicekParams,
    r0_20: float,
    r0_30: float,
    horizon_days: int,
    n_paths: int = 10000,
    trading_days_per_year: float = 252.0,
    random_seed: Optional[int] = None
) -> SimulationResult:
    """
    Run a complete yield simulation.
    
    Parameters
    ----------
    params : BivariateVasicekParams
        Calibrated model parameters.
    r0_20, r0_30 : float
        Current 20Y and 30Y yields.
    horizon_days : int
        Simulation horizon in trading days.
    n_paths : int
        Number of Monte Carlo paths.
    trading_days_per_year : float
        For annualization.
    random_seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    SimulationResult
        Complete simulation results.
    """
    dt = 1.0 / trading_days_per_year
    
    paths_20, paths_30, paths_avg = simulate_average_yield_paths(
        params, r0_20, r0_30, horizon_days, dt, n_paths, random_seed
    )
    
    return SimulationResult(
        paths_20y=paths_20,
        paths_30y=paths_30,
        paths_avg=paths_avg,
        n_steps=horizon_days,
        n_paths=n_paths,
        dt=dt,
        initial_20y=r0_20,
        initial_30y=r0_30
    )


def compute_simulation_statistics(result: SimulationResult) -> dict:
    """
    Compute summary statistics from simulation.
    
    Parameters
    ----------
    result : SimulationResult
        Simulation output.
        
    Returns
    -------
    dict
        Summary statistics.
    """
    terminal_20 = result.terminal_distribution('20y')
    terminal_30 = result.terminal_distribution('30y')
    terminal_avg = result.terminal_distribution('avg')
    
    change_avg = result.yield_change_distribution('avg')
    
    stats = {
        'horizon_days': result.n_steps,
        'n_paths': result.n_paths,
        
        '20y_terminal': {
            'mean': np.mean(terminal_20),
            'std': np.std(terminal_20),
            'p5': np.percentile(terminal_20, 5),
            'p95': np.percentile(terminal_20, 95)
        },
        
        '30y_terminal': {
            'mean': np.mean(terminal_30),
            'std': np.std(terminal_30),
            'p5': np.percentile(terminal_30, 5),
            'p95': np.percentile(terminal_30, 95)
        },
        
        'avg_terminal': {
            'mean': np.mean(terminal_avg),
            'std': np.std(terminal_avg),
            'p5': np.percentile(terminal_avg, 5),
            'p95': np.percentile(terminal_avg, 95)
        },
        
        'avg_yield_change': {
            'mean': np.mean(change_avg),
            'std': np.std(change_avg),
            'p5': np.percentile(change_avg, 5),
            'p95': np.percentile(change_avg, 95),
            'prob_up': np.mean(change_avg > 0),
            'prob_down': np.mean(change_avg < 0)
        }
    }
    
    return stats


def print_simulation_summary(result: SimulationResult) -> None:
    """Print simulation summary."""
    stats = compute_simulation_statistics(result)
    
    initial_avg = (result.initial_20y + result.initial_30y) / 2
    
    print("=" * 60)
    print("YIELD SIMULATION SUMMARY")
    print("=" * 60)
    
    print(f"\nSimulation Parameters:")
    print(f"  Horizon:           {result.n_steps} trading days")
    print(f"  Paths:             {result.n_paths:,}")
    print(f"  Initial 20Y:       {result.initial_20y*100:.3f}%")
    print(f"  Initial 30Y:       {result.initial_30y*100:.3f}%")
    print(f"  Initial Avg:       {initial_avg*100:.3f}%")
    
    print(f"\nTerminal Average Yield Distribution:")
    avg = stats['avg_terminal']
    print(f"  Mean:              {avg['mean']*100:.3f}%")
    print(f"  Std Dev:           {avg['std']*100:.3f}%")
    print(f"  5th percentile:    {avg['p5']*100:.3f}%")
    print(f"  95th percentile:   {avg['p95']*100:.3f}%")
    
    print(f"\nYield Change Distribution (Avg):")
    chg = stats['avg_yield_change']
    print(f"  Mean change:       {chg['mean']*100:+.3f}%")
    print(f"  Std Dev:           {chg['std']*100:.3f}%")
    print(f"  5th percentile:    {chg['p5']*100:+.3f}%")
    print(f"  95th percentile:   {chg['p95']*100:+.3f}%")
    print(f"  P(yields up):      {chg['prob_up']*100:.1f}%")
    print(f"  P(yields down):    {chg['prob_down']*100:.1f}%")
    
    print("=" * 60)
