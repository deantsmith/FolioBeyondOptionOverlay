"""
Exit Parameter Optimization

This module implements grid search optimization over exit rule parameters
to maximize expected return subject to risk constraints.
"""

import numpy as np
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from pricing.exits import (
    ExitRules, ExitReason, SimulationBatchResult,
    simulate_spread_batch
)
from pricing.spreads import SpreadDefinition, SpreadQuote
from simulation.tlt_paths import TLTSimulationResult


@dataclass
class ExitParameterSet:
    """A set of exit rule parameters to evaluate."""
    profit_target_pct: float
    loss_multiple: float
    dte_threshold: int

    def to_exit_rules(self) -> ExitRules:
        """Convert to ExitRules object."""
        return ExitRules(
            profit_target_pct=self.profit_target_pct,
            use_profit_target=True,
            loss_multiple=self.loss_multiple,
            use_loss_multiple=True,
            dte_threshold=self.dte_threshold,
            use_dte_threshold=True
        )

    def __repr__(self):
        return (f"ExitParams(profit={self.profit_target_pct:.0%}, "
                f"loss={self.loss_multiple:.1f}x, dte={self.dte_threshold}d)")


@dataclass
class ParameterEvaluationResult:
    """Results for a single parameter combination."""
    params: ExitParameterSet
    expected_return: float
    cvar_95: float
    pop: float
    risk_adjusted_return: float

    # Statistics
    mean_pnl: float
    std_pnl: float
    var_95: float

    # Exit distribution
    exit_reason_pcts: Dict[str, float]
    avg_days_held: float

    # Constraint status
    passes_constraints: bool
    constraint_violations: List[str]

    def __repr__(self):
        status = "✓" if self.passes_constraints else "✗"
        return (f"{status} {self.params}: E[R]=${self.expected_return:.2f}, "
                f"POP={self.pop:.1%}, CVaR=${self.cvar_95:.2f}")


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""

    # Parameter ranges to search
    profit_target_pcts: List[float] = field(
        default_factory=lambda: [0.30, 0.40, 0.50, 0.60, 0.70]
    )
    loss_multiples: List[float] = field(
        default_factory=lambda: [1.5, 2.0, 2.5, 3.0]
    )
    dte_thresholds: List[int] = field(
        default_factory=lambda: [3, 5, 7, 10, 14]
    )

    # Constraints
    min_pop: float = 0.70
    max_cvar_loss: Optional[float] = None  # Set from risk budget

    # Spread parameters (what to optimize for)
    spread_definition: Optional[SpreadDefinition] = None
    spread_quote: Optional[SpreadQuote] = None

    # Simulation settings
    n_paths: int = 10000
    random_seed: Optional[int] = 42

    # Output
    ranking_metric: str = "expected_return"  # or "risk_adjusted_return"


@dataclass
class OptimizationResult:
    """Complete optimization results."""

    # Input context
    spread: SpreadDefinition
    initial_credit: float
    config: OptimizationConfig

    # All evaluations
    all_results: List[ParameterEvaluationResult]

    # Best result
    best_params: ExitParameterSet
    best_result: ParameterEvaluationResult

    # Statistics
    n_evaluated: int
    n_passing: int
    evaluation_time_seconds: float


class ExitParameterOptimizer:
    """
    Optimizer for exit rule parameters.

    Usage:
        model = CalibratedModel.from_json('calibration.json')
        evaluator = StrategyEvaluator(model)
        evaluator.run_simulations()  # Generate TLT paths

        # Get a candidate spread to optimize
        candidates = evaluator.generate_candidates()
        best_spread = candidates[0]

        optimizer = ExitParameterOptimizer(
            tlt_simulation=evaluator.tlt_simulation,
            config=OptimizationConfig(
                spread_quote=best_spread,
                max_cvar_loss=0.02375
            )
        )

        result = optimizer.run_optimization()
        optimizer.print_results(result)
    """

    def __init__(
        self,
        tlt_simulation: TLTSimulationResult,
        config: OptimizationConfig,
        risk_free_rate: float = 0.05
    ):
        self.tlt_simulation = tlt_simulation
        self.config = config
        self.r = risk_free_rate

    def generate_parameter_grid(self) -> List[ExitParameterSet]:
        """Generate all parameter combinations to evaluate."""
        grid = []
        for profit_pct in self.config.profit_target_pcts:
            for loss_mult in self.config.loss_multiples:
                for dte in self.config.dte_thresholds:
                    grid.append(ExitParameterSet(
                        profit_target_pct=profit_pct,
                        loss_multiple=loss_mult,
                        dte_threshold=dte
                    ))
        return grid

    def evaluate_parameter_set(
        self,
        params: ExitParameterSet,
        definition: SpreadDefinition,
        initial_credit: float
    ) -> ParameterEvaluationResult:
        """Evaluate a single parameter combination."""

        # Convert to exit rules
        exit_rules = params.to_exit_rules()

        # Run batch simulation
        batch_result = simulate_spread_batch(
            definition=definition,
            initial_credit=initial_credit,
            price_paths=self.tlt_simulation.tlt_paths,
            iv_paths=self.tlt_simulation.iv_paths,
            r=self.r,
            rules=exit_rules,
            use_fast=True
        )

        # Extract statistics
        stats = batch_result.stats

        # Check constraints
        violations = []
        if stats['pop'] < self.config.min_pop:
            violations.append(
                f"POP {stats['pop']:.1%} < {self.config.min_pop:.1%}"
            )
        if self.config.max_cvar_loss is not None:
            if stats['cvar_95'] < -self.config.max_cvar_loss:
                violations.append(
                    f"CVaR ${stats['cvar_95']:.2f} exceeds "
                    f"budget ${-self.config.max_cvar_loss:.2f}"
                )

        # Calculate risk-adjusted return
        if abs(stats['cvar_95']) > 0.01:
            rar = stats['expected_value'] / abs(stats['cvar_95'])
        else:
            rar = float('inf') if stats['expected_value'] > 0 else 0

        return ParameterEvaluationResult(
            params=params,
            expected_return=stats['expected_value'],
            cvar_95=stats['cvar_95'],
            pop=stats['pop'],
            risk_adjusted_return=rar,
            mean_pnl=stats['mean_pnl'],
            std_pnl=stats['std_pnl'],
            var_95=stats['var_95'],
            exit_reason_pcts=stats['exit_reason_pcts'],
            avg_days_held=stats['avg_days_held'],
            passes_constraints=len(violations) == 0,
            constraint_violations=violations
        )

    def run_optimization(self, verbose: bool = True) -> OptimizationResult:
        """
        Run complete parameter optimization.

        Returns
        -------
        OptimizationResult
            Complete optimization results with best parameters.
        """

        # Get spread to optimize
        if self.config.spread_quote:
            definition = self.config.spread_quote.definition
            initial_credit = self.config.spread_quote.net_credit
        elif self.config.spread_definition:
            definition = self.config.spread_definition
            # Need to price it
            raise NotImplementedError("Must provide spread_quote with pricing")
        else:
            raise ValueError("Must provide spread_quote or spread_definition")

        if verbose:
            print("\n" + "=" * 70)
            print("EXIT PARAMETER OPTIMIZATION")
            print("=" * 70)
            print(f"\nSpread: {definition}")
            print(f"Initial Credit: ${initial_credit:.2f}")
            print(f"\nConstraints:")
            print(f"  Min POP: {self.config.min_pop:.0%}")
            if self.config.max_cvar_loss:
                print(f"  Max CVaR Loss: ${self.config.max_cvar_loss:.2f}")

        # Generate parameter grid
        param_grid = self.generate_parameter_grid()

        if verbose:
            print(f"\nEvaluating {len(param_grid)} parameter combinations...")
            print(f"  Profit targets: {self.config.profit_target_pcts}")
            print(f"  Loss multiples: {self.config.loss_multiples}")
            print(f"  DTE thresholds: {self.config.dte_thresholds}")

        # Evaluate all combinations
        start_time = time.time()
        results = []

        for i, params in enumerate(param_grid):
            result = self.evaluate_parameter_set(params, definition, initial_credit)
            results.append(result)

            if verbose and (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(param_grid)} combinations")

        elapsed = time.time() - start_time

        # Filter to passing constraints
        passing = [r for r in results if r.passes_constraints]

        if not passing:
            if verbose:
                print("\n⚠️  WARNING: No parameter sets pass all constraints!")
                print("   Showing best unconstrained results instead.")
            # Still find "best" from all results
            passing = results

        # Rank by metric
        if self.config.ranking_metric == "expected_return":
            passing.sort(key=lambda r: r.expected_return, reverse=True)
        else:  # risk_adjusted_return
            passing.sort(key=lambda r: r.risk_adjusted_return, reverse=True)

        best = passing[0]

        if verbose:
            n_passing = sum(1 for r in results if r.passes_constraints)
            print(f"\nOptimization complete in {elapsed:.1f}s")
            print(f"  {n_passing}/{len(results)} parameter sets pass constraints")

        return OptimizationResult(
            spread=definition,
            initial_credit=initial_credit,
            config=self.config,
            all_results=results,
            best_params=best.params,
            best_result=best,
            n_evaluated=len(results),
            n_passing=sum(1 for r in results if r.passes_constraints),
            evaluation_time_seconds=elapsed
        )

    def print_results(
        self,
        result: OptimizationResult,
        top_n: int = 10
    ):
        """Print optimization results."""

        print("\n" + "=" * 90)
        print("OPTIMIZATION RESULTS")
        print("=" * 90)

        print(f"\nBest Parameters:")
        print(f"  Profit Target: {result.best_params.profit_target_pct:.0%}")
        print(f"  Loss Multiple: {result.best_params.loss_multiple:.1f}x")
        print(f"  DTE Threshold: {result.best_params.dte_threshold} days")

        print(f"\nPerformance:")
        print(f"  Expected Return: ${result.best_result.expected_return:.2f}")
        print(f"  POP: {result.best_result.pop:.1%}")
        print(f"  CVaR (95%): ${result.best_result.cvar_95:.2f}")
        print(f"  Risk-Adj Return: {result.best_result.risk_adjusted_return:.2f}")
        print(f"  Avg Days Held: {result.best_result.avg_days_held:.1f}")

        print(f"\nExit Reason Distribution:")
        for reason, pct in result.best_result.exit_reason_pcts.items():
            if pct > 0:
                print(f"  {reason:<20} {pct:.1%}")

        # Show top N
        passing = [r for r in result.all_results if r.passes_constraints]
        if not passing:
            # If none pass, show all sorted by metric
            passing = result.all_results

        passing.sort(
            key=lambda r: r.expected_return
            if result.config.ranking_metric == "expected_return"
            else r.risk_adjusted_return,
            reverse=True
        )

        if len(passing) > 1:
            print(f"\n\nTop {min(top_n, len(passing))} Parameter Sets:")
            print(f"{'Rank':<6} {'Profit%':<10} {'LossMult':<10} {'DTE':<8} "
                  f"{'E[R]':<10} {'POP':<8} {'CVaR95':<10} {'RAR':<8}")
            print("-" * 90)

            for i, r in enumerate(passing[:top_n]):
                print(f"{i+1:<6} {r.params.profit_target_pct:<9.0%} "
                      f"{r.params.loss_multiple:<10.1f} {r.params.dte_threshold:<8} "
                      f"${r.expected_return:<9.2f} {r.pop:<7.1%} "
                      f"${r.cvar_95:<9.2f} {r.risk_adjusted_return:<7.2f}")

        print("=" * 90)


def save_optimization_result(result: OptimizationResult, filepath: str):
    """Save optimization results to JSON."""

    data = {
        'spread': {
            'type': result.spread.spread_type.value,
            'short_strike': result.spread.short_strike,
            'long_strike': result.spread.long_strike,
            'expiration_days': result.spread.expiration_days,
            'is_put_spread': result.spread.is_put_spread
        },
        'initial_credit': result.initial_credit,
        'best_params': {
            'profit_target_pct': result.best_params.profit_target_pct,
            'loss_multiple': result.best_params.loss_multiple,
            'dte_threshold': result.best_params.dte_threshold
        },
        'best_result': {
            'expected_return': result.best_result.expected_return,
            'cvar_95': result.best_result.cvar_95,
            'pop': result.best_result.pop,
            'risk_adjusted_return': result.best_result.risk_adjusted_return,
            'avg_days_held': result.best_result.avg_days_held,
            'exit_reason_pcts': result.best_result.exit_reason_pcts
        },
        'all_results': [
            {
                'params': {
                    'profit_target_pct': r.params.profit_target_pct,
                    'loss_multiple': r.params.loss_multiple,
                    'dte_threshold': r.params.dte_threshold
                },
                'expected_return': r.expected_return,
                'cvar_95': r.cvar_95,
                'pop': r.pop,
                'risk_adjusted_return': r.risk_adjusted_return,
                'passes_constraints': r.passes_constraints
            }
            for r in result.all_results
        ],
        'n_evaluated': result.n_evaluated,
        'n_passing': result.n_passing,
        'evaluation_time_seconds': result.evaluation_time_seconds
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def load_optimization_result(filepath: str) -> Dict:
    """Load optimization results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)
