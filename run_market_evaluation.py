"""
Market-Integrated Strategy Evaluation

This script evaluates option strategies using:
1. Actual market IV from Bloomberg for simulation
2. Real bid/ask prices for spread entry
3. Simulated paths for exit modeling

Usage:
    python run_market_evaluation.py \
        --calibration calibration_results.json \
        --options tlt_options.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import date, timedelta
import numpy as np

# Add script directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.bloomberg import (
    load_bloomberg_chain,
    IVSurface,
    generate_market_spread_candidates,
    print_chain_summary,
    print_spread_candidates,
    MarketSpreadQuote
)
from strategy.evaluation import CalibratedModel, EvaluationConfig
from simulation import run_simulation, simulate_tlt_paths
from pricing import (
    SpreadType, SpreadDefinition,
    ExitRules, simulate_spread_batch,
    print_simulation_report
)
from pricing.spreads import create_spread_by_strikes


def create_market_iv_paths(
    tlt_paths: np.ndarray,
    iv_surface: IVSurface,
    expiration_date: date,
    option_type: str,
    strike: float,
    start_dte: int
) -> np.ndarray:
    """
    Create IV paths using market IV surface.
    
    Adjusts IV based on price movement relative to starting price,
    using the surface's structure to inform the adjustment.
    
    Parameters
    ----------
    tlt_paths : np.ndarray
        Simulated TLT price paths (n_steps+1, n_paths).
    iv_surface : IVSurface
        Market IV surface.
    expiration_date : date
        Option expiration.
    option_type : str
        'put' or 'call'.
    strike : float
        Option strike.
    start_dte : int
        Starting DTE.
        
    Returns
    -------
    np.ndarray
        IV paths matching tlt_paths shape.
    """
    n_steps, n_paths = tlt_paths.shape[0] - 1, tlt_paths.shape[1]
    initial_price = tlt_paths[0, 0]
    
    # Get base IV from surface
    base_iv = iv_surface.get_iv(option_type, expiration_date, strike)
    
    # Get ATM IV for reference
    atm_iv = iv_surface.get_atm_iv(expiration_date)
    
    # Estimate IV-price sensitivity from surface
    # Look at how IV changes across strikes
    strike_range = [strike * 0.95, strike, strike * 1.05]
    ivs_at_strikes = [iv_surface.get_iv(option_type, expiration_date, s) for s in strike_range]
    
    # Approximate sensitivity: dIV/d(price%) 
    # For puts: IV typically rises when price falls
    # For calls: IV typically falls when price rises
    if option_type == 'put':
        price_sensitivity = -(ivs_at_strikes[0] - ivs_at_strikes[2]) / 0.10
    else:
        price_sensitivity = (ivs_at_strikes[2] - ivs_at_strikes[0]) / 0.10
    
    # Build IV paths
    iv_paths = np.zeros_like(tlt_paths)
    
    for step in range(n_steps + 1):
        current_dte = start_dte - step
        if current_dte <= 0:
            current_dte = 1
        
        # Price change from initial
        price_change_pct = (tlt_paths[step, :] - initial_price) / initial_price
        
        # Adjust IV based on price movement
        iv_adjustment = price_sensitivity * price_change_pct
        
        # Term structure adjustment (IV tends toward ATM as expiration approaches)
        term_weight = current_dte / start_dte
        
        # Combine
        iv_paths[step, :] = base_iv + iv_adjustment
        iv_paths[step, :] = iv_paths[step, :] * term_weight + atm_iv * (1 - term_weight) * 0.3
        
        # Floor IV
        iv_paths[step, :] = np.maximum(iv_paths[step, :], 0.05)
    
    return iv_paths


def evaluate_market_spread(
    spread: MarketSpreadQuote,
    model: CalibratedModel,
    iv_surface: IVSurface,
    n_paths: int = 10000,
    exit_rules: ExitRules = None,
    random_seed: int = 42,
    verbose: bool = False
) -> dict:
    """
    Evaluate a market-priced spread using simulation.
    
    Parameters
    ----------
    spread : MarketSpreadQuote
        Spread with market pricing.
    model : CalibratedModel
        Calibrated rate/TLT model.
    iv_surface : IVSurface
        Market IV surface.
    n_paths : int
        Number of simulation paths.
    exit_rules : ExitRules
        Exit rules for the strategy.
    random_seed : int
        Random seed.
    verbose : bool
        Print detailed output.
        
    Returns
    -------
    dict
        Evaluation results.
    """
    if exit_rules is None:
        exit_rules = ExitRules()
    
    # Run rate simulation
    current = model.current_values
    rate_sim = run_simulation(
        params=model.vasicek_params,
        r0_20=current['yield_20y'],
        r0_30=current['yield_30y'],
        horizon_days=spread.dte,
        n_paths=n_paths,
        random_seed=random_seed
    )
    
    # Convert to TLT paths (using model regression)
    tlt_sim = simulate_tlt_paths(
        rate_sim,
        model.regression_params,
        model.volatility_params
    )
    
    # Create market-informed IV paths
    iv_paths = create_market_iv_paths(
        tlt_sim.tlt_paths,
        iv_surface,
        spread.expiration_date,
        spread.option_type,
        spread.short_strike,
        spread.dte
    )
    
    # Create spread definition for simulation
    spread_type = SpreadType.BULL_PUT if spread.option_type == 'put' else SpreadType.BEAR_CALL
    
    spread_def = create_spread_by_strikes(
        spread.short_strike,
        spread.long_strike,
        spread.dte,
        spread_type
    )
    
    # Use actual credit at bid (realistic fill)
    initial_credit = spread.credit_at_bid
    
    # Run simulation
    batch_result = simulate_spread_batch(
        definition=spread_def,
        initial_credit=initial_credit,
        price_paths=tlt_sim.tlt_paths,
        iv_paths=iv_paths,
        r=0.05,  # Risk-free rate
        rules=exit_rules,
        use_fast=True
    )
    
    stats = batch_result.stats
    
    results = {
        'spread': spread,
        'initial_credit': initial_credit,
        'max_loss': spread.max_loss_at_bid,
        'n_paths': n_paths,
        'pop': stats['pop'],
        'expected_return': stats['expected_value'],
        'std_pnl': stats['std_pnl'],
        'cvar_95': stats['cvar_95'],
        'min_pnl': stats['min_pnl'],
        'max_pnl': stats['max_pnl'],
        'avg_days_held': stats['avg_days_held'],
        'exit_reasons': stats['exit_reason_pcts'],
        'batch_result': batch_result
    }
    
    # Risk-adjusted return
    if abs(stats['cvar_95']) > 0.01:
        results['risk_adjusted_return'] = stats['expected_value'] / abs(stats['cvar_95'])
    else:
        results['risk_adjusted_return'] = float('inf') if stats['expected_value'] > 0 else 0
    
    if verbose:
        print(f"\n{spread}")
        print(f"  Credit (at bid): ${initial_credit:.2f}")
        print(f"  POP: {stats['pop']:.1%}")
        print(f"  Expected Return: ${stats['expected_value']:.2f}")
        print(f"  CVaR (95%): ${stats['cvar_95']:.2f}")
        print(f"  Risk-Adj Return: {results['risk_adjusted_return']:.2f}")
    
    return results


def run_market_evaluation(
    calibration_file: str,
    options_file: str,
    n_paths: int = 10000,
    min_dte: int = 30,
    max_dte: int = 90,
    min_pop: float = 0.70,
    delta_targets: list = None,
    spread_widths: list = None,
    exit_rules: ExitRules = None,
    random_seed: int = 42,
    top_n: int = 10,
    verbose: bool = True
):
    """
    Run complete market-integrated evaluation.
    
    Parameters
    ----------
    calibration_file : str
        Path to calibration JSON.
    options_file : str
        Path to Bloomberg options CSV.
    n_paths : int
        Monte Carlo paths.
    min_dte, max_dte : int
        DTE range.
    min_pop : float
        Minimum probability of profit.
    delta_targets : list
        Target deltas for short strikes.
    spread_widths : list
        Spread widths to consider.
    exit_rules : ExitRules
        Exit rules.
    random_seed : int
        Random seed.
    top_n : int
        Number of top strategies to show.
    verbose : bool
        Print progress.
    """
    if delta_targets is None:
        delta_targets = [0.10, 0.15, 0.20, 0.25, 0.30]
    if spread_widths is None:
        spread_widths = [2, 3, 5]
    if exit_rules is None:
        exit_rules = ExitRules()
    
    # Load calibrated model
    if verbose:
        print("=" * 70)
        print("MARKET-INTEGRATED STRATEGY EVALUATION")
        print("=" * 70)
        print(f"\nLoading calibrated model from {calibration_file}...")
    
    model = CalibratedModel.from_json(calibration_file)
    
    if verbose:
        print(f"  Current TLT (model): ${model.current_values['tlt_price']:.2f}")
    
    # Load option chain
    if verbose:
        print(f"\nLoading option chain from {options_file}...")
    
    chain = load_bloomberg_chain(options_file, iv_as_percentage=True)
    
    if verbose:
        print_chain_summary(chain)
    
    # Build IV surface
    iv_surface = IVSurface.from_chain(chain)
    
    # Generate spread candidates from market data
    if verbose:
        print("\nGenerating spread candidates from market data...")
    
    candidates = generate_market_spread_candidates(
        chain,
        min_dte=min_dte,
        max_dte=max_dte,
        delta_targets=delta_targets,
        spread_widths=spread_widths,
        option_types=['put', 'call'],
        min_credit=0.20,
        min_open_interest=100
    )
    
    if verbose:
        print(f"  Found {len(candidates)} valid spread candidates")
        print_spread_candidates(candidates[:20])  # Show first 20
    
    if not candidates:
        print("\nNo valid spread candidates found!")
        return []
    
    # Evaluate each candidate
    if verbose:
        print(f"\nEvaluating candidates with {n_paths:,} Monte Carlo paths...")
    
    results = []
    for i, spread in enumerate(candidates):
        result = evaluate_market_spread(
            spread, model, iv_surface,
            n_paths=n_paths,
            exit_rules=exit_rules,
            random_seed=random_seed,
            verbose=False
        )
        results.append(result)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(candidates)}")
    
    # Filter by POP and rank
    passing = [r for r in results if r['pop'] >= min_pop]
    passing.sort(key=lambda r: r['risk_adjusted_return'], reverse=True)
    
    if verbose:
        print(f"\n{len(passing)}/{len(results)} candidates pass POP >= {min_pop:.0%}")
    
    # Print top results
    if verbose and passing:
        print("\n" + "=" * 100)
        print("TOP STRATEGIES (MARKET-PRICED)")
        print("=" * 100)
        
        print(f"\n{'Rank':<6} {'Type':<6} {'Strikes':<12} {'DTE':<6} "
              f"{'Credit':<10} {'POP':<8} {'E[R]':<10} {'CVaR95':<10} {'RAR':<8}")
        print("-" * 100)
        
        for i, r in enumerate(passing[:top_n]):
            spread = r['spread']
            type_str = "Put" if spread.option_type == 'put' else "Call"
            strikes = f"{spread.short_strike:.0f}/{spread.long_strike:.0f}"
            
            print(f"{i+1:<6} {type_str:<6} {strikes:<12} {spread.dte:<6} "
                  f"${r['initial_credit']:<9.2f} {r['pop']:<7.1%} "
                  f"${r['expected_return']:<9.2f} ${r['cvar_95']:<9.2f} "
                  f"{r['risk_adjusted_return']:<7.2f}")
        
        print("=" * 100)
        
        # Show details of top strategy
        if passing:
            best = passing[0]
            spread = best['spread']
            
            print(f"\nTop Strategy Details:")
            print(f"  {spread}")
            print(f"  Credit at bid: ${best['initial_credit']:.2f}")
            print(f"  Credit at mid: ${spread.credit_at_mid:.2f}")
            print(f"  Max Loss: ${best['max_loss']:.2f}")
            print(f"  Net Delta: {spread.net_delta:.3f}")
            if spread.net_theta:
                print(f"  Net Theta: ${spread.net_theta:.3f}/day")
            print(f"  Avg Days Held: {best['avg_days_held']:.1f}")
            
            print(f"\n  Exit Distribution:")
            for reason, pct in best['exit_reasons'].items():
                if pct > 0:
                    print(f"    {reason:<20} {pct:.1%}")
    
    return passing


def main():
    parser = argparse.ArgumentParser(
        description='Market-Integrated Strategy Evaluation'
    )
    parser.add_argument(
        '--calibration', '-c',
        required=True,
        help='Path to calibration JSON file'
    )
    parser.add_argument(
        '--options', '-o',
        required=True,
        help='Path to Bloomberg options CSV file'
    )
    parser.add_argument(
        '--n-paths', '-n',
        type=int,
        default=10000,
        help='Number of Monte Carlo paths (default: 10000)'
    )
    parser.add_argument(
        '--min-dte',
        type=int,
        default=30,
        help='Minimum DTE (default: 30)'
    )
    parser.add_argument(
        '--max-dte',
        type=int,
        default=90,
        help='Maximum DTE (default: 90)'
    )
    parser.add_argument(
        '--min-pop',
        type=float,
        default=0.70,
        help='Minimum probability of profit (default: 0.70)'
    )
    parser.add_argument(
        '--profit-target',
        type=float,
        default=0.50,
        help='Profit target as fraction of max (default: 0.50)'
    )
    parser.add_argument(
        '--loss-multiple',
        type=float,
        default=2.0,
        help='Close when spread value reaches this multiple of credit (default: 2.0)'
    )
    parser.add_argument(
        '--dte-close',
        type=int,
        default=7,
        help='Days before expiration to close (default: 7)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top strategies to display'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    exit_rules = ExitRules(
        profit_target_pct=args.profit_target,
        use_profit_target=True,
        loss_multiple=args.loss_multiple,
        use_loss_multiple=True,
        dte_threshold=args.dte_close,
        use_dte_threshold=True
    )
    
    try:
        results = run_market_evaluation(
            calibration_file=args.calibration,
            options_file=args.options,
            n_paths=args.n_paths,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
            min_pop=args.min_pop,
            exit_rules=exit_rules,
            random_seed=args.seed,
            top_n=args.top,
            verbose=not args.quiet
        )
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
