"""
Strategy Evaluation Runner

This script runs the complete strategy evaluation pipeline:
1. Load calibrated model
2. Generate spread candidates
3. Run Monte Carlo simulations
4. Evaluate strategies with exit rules
5. Rank by risk-adjusted return

Usage:
    python run_strategy_evaluation.py --calibration calibration_results.json
"""

import argparse
import sys
from pathlib import Path

# Add script directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategy.evaluation import (
    CalibratedModel,
    EvaluationConfig,
    StrategyEvaluator,
    run_evaluation
)
from pricing.exits import ExitRules
from pricing.spreads import SpreadType


def main():
    parser = argparse.ArgumentParser(
        description='Run TLT Options Strategy Evaluation'
    )
    parser.add_argument(
        '--calibration', '-c',
        required=True,
        help='Path to calibration JSON file'
    )
    parser.add_argument(
        '--n-paths', '-n',
        type=int,
        default=10000,
        help='Number of Monte Carlo paths (default: 10000)'
    )
    parser.add_argument(
        '--expiration', '-e',
        type=int,
        default=45,
        help='Days to expiration (default: 45)'
    )
    parser.add_argument(
        '--min-pop',
        type=float,
        default=0.70,
        help='Minimum probability of profit (default: 0.70)'
    )
    parser.add_argument(
        '--max-cvar-loss',
        type=float,
        default=None,
        help='Maximum acceptable CVaR loss in dollars (optional)'
    )
    parser.add_argument(
        '--profit-target',
        type=float,
        default=0.50,
        help='Profit target as fraction of max profit (default: 0.50)'
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
        '--put-only',
        action='store_true',
        help='Only evaluate put spreads'
    )
    parser.add_argument(
        '--call-only',
        action='store_true',
        help='Only evaluate call spreads'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top strategies to display (default: 10)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Determine spread types
    if args.put_only:
        spread_types = [SpreadType.BULL_PUT]
    elif args.call_only:
        spread_types = [SpreadType.BEAR_CALL]
    else:
        spread_types = [SpreadType.BULL_PUT, SpreadType.BEAR_CALL]
    
    # Build exit rules
    exit_rules = ExitRules(
        profit_target_pct=args.profit_target,
        use_profit_target=True,
        loss_multiple=args.loss_multiple,
        use_loss_multiple=True,
        dte_threshold=args.dte_close,
        use_dte_threshold=True
    )
    
    # Build evaluation config
    config = EvaluationConfig(
        n_paths=args.n_paths,
        random_seed=args.seed,
        expiration_days=args.expiration,
        delta_targets=[0.10, 0.15, 0.20, 0.25, 0.30],
        spread_widths=[2, 3, 5],
        spread_types=spread_types,
        min_pop=args.min_pop,
        max_cvar_loss=args.max_cvar_loss,
        exit_rules=exit_rules
    )
    
    # Run evaluation
    verbose = not args.quiet
    
    if verbose:
        print("\n" + "=" * 70)
        print("TLT OPTIONS OVERLAY - STRATEGY EVALUATION")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Calibration file: {args.calibration}")
        print(f"  Monte Carlo paths: {args.n_paths:,}")
        print(f"  Expiration days: {args.expiration}")
        print(f"  Min POP: {args.min_pop:.0%}")
        if args.max_cvar_loss is not None:
            print(f"  Max CVaR Loss: ${args.max_cvar_loss:.2f}")
        print(f"\nExit Rules:")
        print(f"  {exit_rules}")
    
    try:
        # Load model
        model = CalibratedModel.from_json(args.calibration)
        
        if verbose:
            print(f"\nModel loaded successfully")
            print(f"  Current TLT: ${model.current_values['tlt_price']:.2f}")
            print(f"  Base IV: {model.volatility_params.base_iv:.1%}")
        
        # Create evaluator and run
        evaluator = StrategyEvaluator(model, config)
        candidates = evaluator.evaluate_strategies(verbose=verbose)
        
        # Print results
        if verbose:
            evaluator.print_results(top_n=args.top)
        
        # Summary
        n_passing = sum(1 for c in candidates if c.passes_constraints)
        
        if verbose:
            print(f"\n{n_passing} strategies pass all constraints")
            
            if n_passing > 0:
                best = candidates[0]
                print(f"\nBest Strategy: {best.spread_quote.definition}")
                print(f"  Expected Return: ${best.expected_return:.2f}")
                print(f"  Probability of Profit: {best.pop:.1%}")
                print(f"  CVaR (95%): ${best.cvar_95:.2f}")
                print(f"  Risk-Adjusted Return: {best.risk_adjusted_return:.2f}")
        
        return 0
        
    except FileNotFoundError:
        print(f"Error: Calibration file not found: {args.calibration}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
