"""
Single-Spread Market Evaluation

This script evaluates a single user-specified credit spread using:
1. Actual market IV from Bloomberg for simulation
2. Real bid/ask prices for spread entry
3. Simulated paths for exit modeling

Usage example:
    python run_single_market_spread.py \
        --calibration calibration_results.json \
        --options tlt_options.csv \
        --type put \
        --short-strike 85 \
        --long-strike 83 \
        --dte 45
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add script directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.bloomberg import (  # noqa: E402
    load_bloomberg_chain,
    IVSurface,
    price_spread_from_chain,
    MarketSpreadQuote,
)
from strategy.evaluation import CalibratedModel  # noqa: E402
from pricing import ExitRules  # noqa: E402

from run_market_evaluation import (  # noqa: E402
    evaluate_single_market_candidate,
)


def _choose_expiration_by_dte(chain, target_dte: int, tolerance: int = 5) -> Optional[object]:
    """
    Choose the expiration whose DTE is closest to target_dte.

    Returns the expiration date object, or None if no expiration is
    within the specified tolerance (in days).
    """
    quote_date = chain.quote_date
    expirations = chain.expirations

    if not expirations:
        return None

    best_exp = None
    best_diff = None

    for exp in expirations:
        dte = (exp - quote_date).days
        diff = abs(dte - target_dte)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_exp = exp

    if best_diff is None or best_diff > tolerance:
        return None

    return best_exp


def print_single_spread_report(result: dict) -> None:
    """Print a report for a single evaluated market spread."""
    spread: MarketSpreadQuote = result["spread"]

    print("\n" + "=" * 130)
    print("SINGLE STRATEGY (MARKET-PRICED)")
    print("=" * 130)

    type_str = "Put" if spread.option_type == "put" else "Call"
    strikes = f"{spread.short_strike:.0f}/{spread.long_strike:.0f}"

    print(f"\nType:        {type_str}")
    print(f"Strikes:     {strikes}")
    print(f"Expiration:  {spread.expiration_date} ({spread.dte} DTE)")
    print(f"Credit at bid: ${result['initial_credit']:.2f}")
    print(f"Credit at mid: ${spread.credit_at_mid:.2f}")
    print(f"Max Loss:      ${result['max_loss']:.2f}")
    print(f"Net Delta:     {spread.net_delta:.3f}")
    if spread.net_theta:
        print(f"Net Theta:    ${spread.net_theta:.3f}/day")

    print("\nSimulation Results:")
    print(f"  POP:               {result['pop']:.1%}")
    print(f"  Expected Return:   ${result['expected_return']:.2f}")
    print(f"  CVaR (95%):        ${result['cvar_95']:.2f}")
    print(f"  Risk-Adj Return:   {result['risk_adjusted_return']:.2f}")
    print(f"  Avg Days Held:     {result['avg_days_held']:.1f}")
    print(f"  Min PnL:           ${result['min_pnl']:.2f}")
    print(f"  Max PnL:           ${result['max_pnl']:.2f}")
    print(f"  Std Dev PnL:       ${result['std_pnl']:.2f}")

    if "passes_constraints" in result:
        status = "PASS" if result["passes_constraints"] else "FAIL"
        print(f"\nConstraint Status: {status}")
        if not result["passes_constraints"]:
            violations = result.get("constraint_violations") or []
            for v in violations:
                print(f"  - {v}")

    print("\nExit Distribution:")
    for reason, pct in result["exit_reasons"].items():
        if pct > 0:
            print(f"  {reason:<20} {pct:.1%}")

    short_contract = spread.short_contract
    print("\nMarket Liquidity (short leg):")

    def _format_int(value):
        if value is None:
            return "-"
        try:
            if np.isnan(value):
                return "-"
        except TypeError:
            pass
        try:
            return f"{int(round(value))}"
        except (TypeError, ValueError):
            return "-"

    print(f"  Volume:      {_format_int(short_contract.volume)}")
    print(f"  OpenInterest:{_format_int(short_contract.open_interest)}")

    print("=" * 130)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a single market-priced credit spread"
    )
    parser.add_argument(
        "--calibration",
        "-c",
        required=True,
        help="Path to calibration JSON file",
    )
    parser.add_argument(
        "--options",
        "-o",
        required=True,
        help="Path to Bloomberg options CSV file",
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["put", "call"],
        required=True,
        help="Spread type: put or call",
    )
    parser.add_argument(
        "--short-strike",
        type=float,
        required=True,
        help="Short leg strike",
    )
    parser.add_argument(
        "--long-strike",
        type=float,
        required=True,
        help="Long leg strike",
    )
    parser.add_argument(
        "--dte",
        type=int,
        required=True,
        help="Target days to expiration",
    )
    parser.add_argument(
        "--n-paths",
        "-n",
        type=int,
        default=10000,
        help="Number of Monte Carlo paths (default: 10000)",
    )
    parser.add_argument(
        "--min-pop",
        type=float,
        default=0.70,
        help="Minimum probability of profit for constraint check (default: 0.70)",
    )
    parser.add_argument(
        "--max-cvar-loss",
        type=float,
        default=None,
        help="Maximum acceptable CVaR loss in dollars (optional, positive number)",
    )
    parser.add_argument(
        "--profit-target",
        type=float,
        default=0.50,
        help="Profit target as fraction of max (default: 0.50)",
    )
    parser.add_argument(
        "--loss-multiple",
        type=float,
        default=2.0,
        help="Close when spread value reaches this multiple of credit (default: 2.0)",
    )
    parser.add_argument(
        "--dte-close",
        type=int,
        default=7,
        help="Days before expiration to close (default: 7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    exit_rules = ExitRules(
        profit_target_pct=args.profit_target,
        use_profit_target=True,
        loss_multiple=args.loss_multiple,
        use_loss_multiple=True,
        dte_threshold=args.dte_close,
        use_dte_threshold=True,
    )

    try:
        if not args.quiet:
            print("=" * 70)
            print("SINGLE-SPREAD MARKET EVALUATION")
            print("=" * 70)
            print(f"\nLoading calibrated model from {args.calibration}...")

        model = CalibratedModel.from_json(args.calibration)

        if not args.quiet:
            print(f"  Current TLT (model): ${model.current_values['tlt_price']:.2f}")
            print(f"\nLoading option chain from {args.options}...")

        chain = load_bloomberg_chain(args.options, iv_as_percentage=True)
        iv_surface = IVSurface.from_chain(chain)

        expiration = _choose_expiration_by_dte(chain, args.dte)
        if expiration is None:
            print(
                f"Error: No expiration within tolerance of target DTE={args.dte} days."
            )
            return 1

        spread = price_spread_from_chain(
            chain=chain,
            option_type=args.type,
            short_strike=args.short_strike,
            long_strike=args.long_strike,
            expiration=expiration,
        )

        if spread is None:
            print(
                "Error: Could not find both short and long contracts for "
                f"{args.type} spread {args.short_strike}/{args.long_strike} "
                f"at expiration {expiration}."
            )
            return 1

        result = evaluate_single_market_candidate(
            spread=spread,
            model=model,
            iv_surface=iv_surface,
            n_paths=args.n_paths,
            exit_rules=exit_rules,
            random_seed=args.seed,
            min_pop=args.min_pop,
            max_cvar_loss=args.max_cvar_loss,
        )

        print_single_spread_report(result)

        return 0

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

