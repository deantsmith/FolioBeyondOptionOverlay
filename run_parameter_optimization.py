"""
Exit Parameter Optimization Runner

This script optimizes exit rule parameters for a given spread strategy:
1. Load calibrated model
2. Generate TLT simulation paths
3. Price the target spread
4. Run grid search over exit parameters
5. Rank by expected return (subject to constraints)

Usage:
    python run_parameter_optimization.py --calibration calibration_results.json \\
        --delta 0.20 --width 3 --spread-type put --expiration 45 \\
        --profit-targets 0.30,0.40,0.50,0.60,0.70 \\
        --loss-multiples 1.5,2.0,2.5,3.0 \\
        --dte-thresholds 3,5,7,10,14 \\
        --output optimal_params.json
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add script directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategy.evaluation import CalibratedModel, StrategyEvaluator
from strategy.optimization import (
    ExitParameterOptimizer,
    OptimizationConfig,
    save_optimization_result
)
from pricing.spreads import SpreadType, create_spread_by_delta, price_spread
from simulation import run_simulation, simulate_tlt_paths


def parse_comma_separated_floats(value: str) -> List[float]:
    """Parse comma-separated float values."""
    try:
        return [float(x.strip()) for x in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float list: {value}")


def parse_comma_separated_ints(value: str) -> List[int]:
    """Parse comma-separated integer values."""
    try:
        return [int(x.strip()) for x in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int list: {value}")


def main():
    parser = argparse.ArgumentParser(
        description='Optimize Exit Rule Parameters for TLT Credit Spreads',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize a 20-delta, 3-point bull put spread
  python run_parameter_optimization.py -c calibration.json \\
      --delta 0.20 --width 3 --spread-type put

  # Specify custom parameter ranges
  python run_parameter_optimization.py -c calibration.json \\
      --delta 0.25 --width 5 --spread-type call \\
      --profit-targets 0.40,0.50,0.60 \\
      --loss-multiples 1.5,2.0,2.5 \\
      --dte-thresholds 5,7,10

  # Set constraints and save results
  python run_parameter_optimization.py -c calibration.json \\
      --delta 0.20 --width 3 --spread-type put \\
      --min-pop 0.70 --risk-budget 0.02375 \\
      --output optimal_params.json
        """
    )

    # Required arguments
    parser.add_argument(
        '--calibration', '-c',
        required=True,
        help='Path to calibration JSON file'
    )

    # Spread specification
    parser.add_argument(
        '--delta',
        type=float,
        required=True,
        help='Target delta for short strike (e.g., 0.20 for 20-delta)'
    )
    parser.add_argument(
        '--width',
        type=float,
        required=True,
        help='Spread width in dollars (e.g., 3.0)'
    )
    parser.add_argument(
        '--spread-type',
        choices=['put', 'call'],
        required=True,
        help='Type of spread: put (bull put) or call (bear call)'
    )
    parser.add_argument(
        '--expiration',
        type=int,
        default=45,
        help='Days to expiration (default: 45)'
    )

    # Parameter ranges to search
    parser.add_argument(
        '--profit-targets',
        type=parse_comma_separated_floats,
        default=[0.30, 0.40, 0.50, 0.60, 0.70],
        help='Profit target percentages to test (comma-separated, default: 0.30,0.40,0.50,0.60,0.70)'
    )
    parser.add_argument(
        '--loss-multiples',
        type=parse_comma_separated_floats,
        default=[1.5, 2.0, 2.5, 3.0],
        help='Loss multiples to test (comma-separated, default: 1.5,2.0,2.5,3.0)'
    )
    parser.add_argument(
        '--dte-thresholds',
        type=parse_comma_separated_ints,
        default=[3, 5, 7, 10, 14],
        help='DTE thresholds to test (comma-separated, default: 3,5,7,10,14)'
    )

    # Constraints
    parser.add_argument(
        '--min-pop',
        type=float,
        default=0.70,
        help='Minimum probability of profit constraint (default: 0.70)'
    )
    parser.add_argument(
        '--risk-budget',
        type=float,
        default=None,
        help='Maximum CVaR loss budget (default: None, no constraint)'
    )

    # Simulation settings
    parser.add_argument(
        '--n-paths',
        type=int,
        default=10000,
        help='Number of Monte Carlo paths (default: 10000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.05,
        help='Risk-free rate (default: 0.05)'
    )

    # Output
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save optimization results JSON (optional)'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top parameter sets to display (default: 10)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--ranking-metric',
        choices=['expected_return', 'risk_adjusted_return'],
        default='expected_return',
        help='Metric to rank parameter sets (default: expected_return)'
    )

    args = parser.parse_args()

    # Load calibrated model
    if not args.quiet:
        print(f"Loading calibration from: {args.calibration}")

    try:
        model = CalibratedModel.from_json(args.calibration)
    except FileNotFoundError:
        print(f"Error: Calibration file not found: {args.calibration}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading calibration: {e}")
        sys.exit(1)

    if not args.quiet:
        print(f"✓ Calibration loaded")
        print(f"\nGenerating {args.n_paths:,} Monte Carlo simulation paths...")

    # Run rate simulation
    rate_simulation = run_simulation(
        params=model.vasicek_params,
        r0_20=model.current_values['yield_20y'],
        r0_30=model.current_values['yield_30y'],
        horizon_days=args.expiration,
        n_paths=args.n_paths,
        random_seed=args.seed
    )

    # Generate TLT paths from rate simulation
    tlt_result = simulate_tlt_paths(
        rate_simulation=rate_simulation,
        regression_params=model.regression_params,
        vol_params=model.volatility_params
    )

    if not args.quiet:
        print(f"✓ Generated {args.n_paths:,} paths for {args.expiration} days")
        print(f"\nPricing target spread...")

    # Create spread
    spread_type = SpreadType.BULL_PUT if args.spread_type == 'put' else SpreadType.BEAR_CALL
    spot = model.current_values['tlt_price']
    sigma = tlt_result.iv_paths[0, 0]  # Initial IV

    spread_definition = create_spread_by_delta(
        spot=spot,
        r=args.risk_free_rate,
        sigma=sigma,
        expiration_days=args.expiration,
        target_delta=args.delta,
        spread_width=args.width,
        spread_type=spread_type
    )

    # Price the spread
    spread_quote = price_spread(
        definition=spread_definition,
        spot=spot,
        r=args.risk_free_rate,
        sigma=sigma
    )

    if not args.quiet:
        print(f"✓ {spread_quote.definition}")
        print(f"  Credit: ${spread_quote.net_credit:.2f}")
        print(f"  Max Profit: ${spread_quote.max_profit:.2f}")
        print(f"  Max Loss: ${spread_quote.max_loss:.2f}")

    # Configure optimization
    config = OptimizationConfig(
        profit_target_pcts=args.profit_targets,
        loss_multiples=args.loss_multiples,
        dte_thresholds=args.dte_thresholds,
        min_pop=args.min_pop,
        max_cvar_loss=args.risk_budget,
        spread_quote=spread_quote,
        n_paths=args.n_paths,
        random_seed=args.seed,
        ranking_metric=args.ranking_metric
    )

    # Run optimization
    optimizer = ExitParameterOptimizer(
        tlt_simulation=tlt_result,
        config=config,
        risk_free_rate=args.risk_free_rate
    )

    result = optimizer.run_optimization(verbose=not args.quiet)

    # Display results
    optimizer.print_results(result, top_n=args.top)

    # Save results if requested
    if args.output:
        save_optimization_result(result, args.output)

    # Return 0 if at least one parameter set passes constraints
    return 0 if result.n_passing > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
