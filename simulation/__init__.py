"""Simulation modules for rate and TLT path generation."""

from .rate_paths import (
    simulate_univariate_vasicek,
    simulate_bivariate_vasicek,
    simulate_average_yield_paths,
    SimulationResult,
    run_simulation,
    compute_simulation_statistics,
    print_simulation_summary
)

from .tlt_paths import (
    TLTSimulationResult,
    convert_yields_to_tlt,
    simulate_tlt_paths,
    compute_tlt_statistics,
    print_tlt_simulation_summary,
    extract_scenario_paths
)

__all__ = [
    # Rate paths
    'simulate_univariate_vasicek',
    'simulate_bivariate_vasicek',
    'simulate_average_yield_paths',
    'SimulationResult',
    'run_simulation',
    'compute_simulation_statistics',
    'print_simulation_summary',
    
    # TLT paths
    'TLTSimulationResult',
    'convert_yields_to_tlt',
    'simulate_tlt_paths',
    'compute_tlt_statistics',
    'print_tlt_simulation_summary',
    'extract_scenario_paths'
]
