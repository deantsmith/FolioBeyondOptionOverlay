"""
Path-Dependent Exit Logic

This module implements exit rules for credit spreads:
1. Profit target (close at X% of max profit)
2. Loss limit (close at X% of max loss)
3. DTE threshold (close N days before expiration)
4. Expiration (intrinsic value settlement)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum

from .spreads import (
    SpreadDefinition, SpreadQuote, SpreadType,
    value_spread, value_spread_at_expiration
)
from .black_scholes import put_price, call_price


class ExitReason(Enum):
    """Reason for exiting a position."""
    PROFIT_TARGET = "profit_target"
    LOSS_LIMIT = "loss_limit"
    DTE_THRESHOLD = "dte_threshold"
    EXPIRATION = "expiration"
    STILL_OPEN = "still_open"


@dataclass
class ExitRules:
    """Configuration for exit rules."""
    
    # Profit target: close when P&L >= X% of max profit
    profit_target_pct: float = 0.50  # 50% of max profit
    use_profit_target: bool = True
    
    # Loss limit: close when P&L <= -X% of max loss (i.e., loss exceeds threshold)
    loss_limit_pct: float = 1.0  # 100% of max loss (i.e., at max loss)
    use_loss_limit: bool = True
    
    # Alternative: close when spread value >= X times credit received
    loss_multiple: Optional[float] = 2.0  # Close at 200% of credit
    use_loss_multiple: bool = True
    
    # DTE threshold: close with N days remaining
    dte_threshold: int = 7
    use_dte_threshold: bool = True
    
    def __repr__(self):
        rules = []
        if self.use_profit_target:
            rules.append(f"profit≥{self.profit_target_pct:.0%}")
        if self.use_loss_multiple:
            rules.append(f"value≥{self.loss_multiple:.1f}x credit")
        elif self.use_loss_limit:
            rules.append(f"loss≥{self.loss_limit_pct:.0%}")
        if self.use_dte_threshold:
            rules.append(f"DTE≤{self.dte_threshold}")
        return f"ExitRules({', '.join(rules)})"


@dataclass
class ExitResult:
    """Result of applying exit rules to a position."""
    
    exit_day: int                    # Day of exit (0 = no exit yet)
    exit_reason: ExitReason
    exit_pnl: float                  # P&L at exit
    exit_price: float                # Underlying price at exit
    days_held: int                   # Days position was held
    
    # For analysis
    max_pnl_seen: float = 0.0        # Best P&L during holding period
    min_pnl_seen: float = 0.0        # Worst P&L during holding period
    
    def __repr__(self):
        return (f"ExitResult({self.exit_reason.value}, day={self.exit_day}, "
                f"P&L=${self.exit_pnl:.2f})")


def check_exit_conditions(
    pnl: float,
    spread_value: float,
    initial_credit: float,
    max_profit: float,
    max_loss: float,
    days_remaining: int,
    rules: ExitRules
) -> Tuple[bool, ExitReason]:
    """
    Check if any exit condition is met.
    
    Parameters
    ----------
    pnl : float
        Current unrealized P&L.
    spread_value : float
        Current cost to close the spread.
    initial_credit : float
        Credit received at open.
    max_profit : float
        Maximum possible profit (= initial credit).
    max_loss : float
        Maximum possible loss.
    days_remaining : int
        Days until expiration.
    rules : ExitRules
        Exit rule configuration.
        
    Returns
    -------
    Tuple[bool, ExitReason]
        (should_exit, reason)
    """
    # Check profit target
    if rules.use_profit_target:
        profit_threshold = max_profit * rules.profit_target_pct
        if pnl >= profit_threshold:
            return True, ExitReason.PROFIT_TARGET
    
    # Check loss limit (using spread value multiple)
    if rules.use_loss_multiple:
        loss_threshold_value = initial_credit * rules.loss_multiple
        if spread_value >= loss_threshold_value:
            return True, ExitReason.LOSS_LIMIT
    elif rules.use_loss_limit:
        loss_threshold = -max_loss * rules.loss_limit_pct
        if pnl <= loss_threshold:
            return True, ExitReason.LOSS_LIMIT
    
    # Check DTE threshold
    if rules.use_dte_threshold:
        if days_remaining <= rules.dte_threshold:
            return True, ExitReason.DTE_THRESHOLD
    
    return False, ExitReason.STILL_OPEN


def simulate_position(
    definition: SpreadDefinition,
    initial_credit: float,
    price_path: np.ndarray,
    iv_path: np.ndarray,
    r: float,
    rules: ExitRules,
    trading_days_per_year: float = 252.0
) -> ExitResult:
    """
    Simulate a spread position through a price path with exit rules.
    
    Parameters
    ----------
    definition : SpreadDefinition
        Spread definition.
    initial_credit : float
        Credit received at open.
    price_path : np.ndarray
        Simulated underlying prices (length = days + 1).
    iv_path : np.ndarray
        Simulated implied volatilities.
    r : float
        Risk-free rate.
    rules : ExitRules
        Exit rules to apply.
    trading_days_per_year : float
        For time conversion.
        
    Returns
    -------
    ExitResult
        Result of the simulation.
    """
    max_profit = initial_credit
    max_loss = definition.width - initial_credit
    
    total_days = definition.expiration_days
    n_steps = len(price_path) - 1
    
    # Track P&L extremes
    max_pnl_seen = 0.0
    min_pnl_seen = 0.0
    
    # Walk through the path
    for day in range(1, n_steps + 1):
        days_remaining = total_days - day
        spot = price_path[day]
        sigma = iv_path[day] if day < len(iv_path) else iv_path[-1]
        
        # Calculate current P&L
        if days_remaining <= 0:
            # At expiration
            pnl = value_spread_at_expiration(definition, spot, initial_credit)
            return ExitResult(
                exit_day=day,
                exit_reason=ExitReason.EXPIRATION,
                exit_pnl=pnl,
                exit_price=spot,
                days_held=day,
                max_pnl_seen=max_pnl_seen,
                min_pnl_seen=min_pnl_seen
            )
        
        # Value the spread
        spread_value, pnl = value_spread(
            definition, spot, r, sigma, days_remaining, initial_credit
        )
        
        # Track extremes
        max_pnl_seen = max(max_pnl_seen, pnl)
        min_pnl_seen = min(min_pnl_seen, pnl)
        
        # Check exit conditions
        should_exit, reason = check_exit_conditions(
            pnl, spread_value, initial_credit, max_profit, max_loss,
            days_remaining, rules
        )
        
        if should_exit:
            return ExitResult(
                exit_day=day,
                exit_reason=reason,
                exit_pnl=pnl,
                exit_price=spot,
                days_held=day,
                max_pnl_seen=max_pnl_seen,
                min_pnl_seen=min_pnl_seen
            )
    
    # If we get here, position expired
    final_spot = price_path[-1]
    final_pnl = value_spread_at_expiration(definition, final_spot, initial_credit)
    
    return ExitResult(
        exit_day=n_steps,
        exit_reason=ExitReason.EXPIRATION,
        exit_pnl=final_pnl,
        exit_price=final_spot,
        days_held=n_steps,
        max_pnl_seen=max_pnl_seen,
        min_pnl_seen=min_pnl_seen
    )


def simulate_position_fast(
    definition: SpreadDefinition,
    initial_credit: float,
    price_path: np.ndarray,
    iv_path: np.ndarray,
    r: float,
    rules: ExitRules
) -> ExitResult:
    """
    Fast simulation that checks only key points.
    
    For efficiency, checks:
    1. DTE threshold day
    2. Expiration
    3. Samples intermediate days for profit/loss triggers
    
    Parameters
    ----------
    definition : SpreadDefinition
        Spread definition.
    initial_credit : float
        Credit received.
    price_path : np.ndarray
        Price path.
    iv_path : np.ndarray
        IV path.
    r : float
        Risk-free rate.
    rules : ExitRules
        Exit rules.
        
    Returns
    -------
    ExitResult
        Simulation result.
    """
    max_profit = initial_credit
    max_loss = definition.width - initial_credit
    total_days = definition.expiration_days
    n_steps = min(len(price_path) - 1, total_days)
    
    # Determine check points
    # Check every day for first week, then every 2-3 days
    check_days = list(range(1, min(8, n_steps + 1)))
    check_days += list(range(8, n_steps + 1, 2))
    
    # Always check DTE threshold day and expiration
    dte_day = total_days - rules.dte_threshold
    if dte_day > 0 and dte_day not in check_days:
        check_days.append(dte_day)
    if n_steps not in check_days:
        check_days.append(n_steps)
    
    check_days = sorted(set(check_days))
    
    max_pnl_seen = 0.0
    min_pnl_seen = 0.0
    
    for day in check_days:
        if day > n_steps:
            break
            
        days_remaining = total_days - day
        spot = price_path[day]
        sigma = iv_path[day] if day < len(iv_path) else iv_path[-1]
        
        if days_remaining <= 0:
            pnl = value_spread_at_expiration(definition, spot, initial_credit)
            return ExitResult(
                exit_day=day,
                exit_reason=ExitReason.EXPIRATION,
                exit_pnl=pnl,
                exit_price=spot,
                days_held=day,
                max_pnl_seen=max(max_pnl_seen, pnl),
                min_pnl_seen=min(min_pnl_seen, pnl)
            )
        
        spread_value, pnl = value_spread(
            definition, spot, r, sigma, days_remaining, initial_credit
        )
        
        max_pnl_seen = max(max_pnl_seen, pnl)
        min_pnl_seen = min(min_pnl_seen, pnl)
        
        should_exit, reason = check_exit_conditions(
            pnl, spread_value, initial_credit, max_profit, max_loss,
            days_remaining, rules
        )
        
        if should_exit:
            return ExitResult(
                exit_day=day,
                exit_reason=reason,
                exit_pnl=pnl,
                exit_price=spot,
                days_held=day,
                max_pnl_seen=max_pnl_seen,
                min_pnl_seen=min_pnl_seen
            )
    
    # Expiration
    final_spot = price_path[min(n_steps, len(price_path) - 1)]
    final_pnl = value_spread_at_expiration(definition, final_spot, initial_credit)
    
    return ExitResult(
        exit_day=n_steps,
        exit_reason=ExitReason.EXPIRATION,
        exit_pnl=final_pnl,
        exit_price=final_spot,
        days_held=n_steps,
        max_pnl_seen=max(max_pnl_seen, final_pnl),
        min_pnl_seen=min(min_pnl_seen, final_pnl)
    )


@dataclass
class SimulationBatchResult:
    """Results from simulating a spread across many paths."""
    
    definition: SpreadDefinition
    initial_credit: float
    n_paths: int
    
    # P&L distribution
    pnls: np.ndarray
    
    # Exit statistics
    exit_reasons: List[ExitReason]
    exit_days: np.ndarray
    
    # Summary statistics (computed on demand)
    _stats: Optional[dict] = None
    
    @property
    def stats(self) -> dict:
        """Compute summary statistics."""
        if self._stats is not None:
            return self._stats
        
        pnls = self.pnls
        
        # Basic P&L stats
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        
        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        pnl_percentiles = {p: np.percentile(pnls, p) for p in percentiles}
        
        # Probability of profit
        pop = np.mean(pnls > 0)
        
        # Expected value
        expected_value = mean_pnl
        
        # CVaR (Expected Shortfall) at 95%
        var_95 = np.percentile(pnls, 5)  # 5th percentile = 95% VaR
        cvar_95 = np.mean(pnls[pnls <= var_95])
        
        # Max metrics
        max_profit = self.initial_credit
        max_loss = self.definition.width - self.initial_credit
        
        # Exit reason breakdown
        reason_counts = {}
        for reason in ExitReason:
            count = sum(1 for r in self.exit_reasons if r == reason)
            reason_counts[reason.value] = count / self.n_paths
        
        # Average days held
        avg_days_held = np.mean(self.exit_days)
        
        self._stats = {
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'median_pnl': pnl_percentiles[50],
            'pnl_percentiles': pnl_percentiles,
            'pop': pop,
            'expected_value': expected_value,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'exit_reason_pcts': reason_counts,
            'avg_days_held': avg_days_held,
            'min_pnl': np.min(pnls),
            'max_pnl': np.max(pnls)
        }
        
        return self._stats


def simulate_spread_batch(
    definition: SpreadDefinition,
    initial_credit: float,
    price_paths: np.ndarray,
    iv_paths: np.ndarray,
    r: float,
    rules: ExitRules,
    use_fast: bool = True
) -> SimulationBatchResult:
    """
    Simulate a spread across multiple price paths.
    
    Parameters
    ----------
    definition : SpreadDefinition
        Spread definition.
    initial_credit : float
        Credit received.
    price_paths : np.ndarray
        Price paths, shape (n_steps + 1, n_paths).
    iv_paths : np.ndarray
        IV paths, same shape.
    r : float
        Risk-free rate.
    rules : ExitRules
        Exit rules.
    use_fast : bool
        Use fast simulation (samples fewer days).
        
    Returns
    -------
    SimulationBatchResult
        Batch simulation results.
    """
    n_paths = price_paths.shape[1]
    
    pnls = np.zeros(n_paths)
    exit_days = np.zeros(n_paths)
    exit_reasons = []
    
    sim_func = simulate_position_fast if use_fast else simulate_position
    
    for i in range(n_paths):
        result = sim_func(
            definition,
            initial_credit,
            price_paths[:, i],
            iv_paths[:, i],
            r,
            rules
        )
        
        pnls[i] = result.exit_pnl
        exit_days[i] = result.exit_day
        exit_reasons.append(result.exit_reason)
    
    return SimulationBatchResult(
        definition=definition,
        initial_credit=initial_credit,
        n_paths=n_paths,
        pnls=pnls,
        exit_reasons=exit_reasons,
        exit_days=exit_days
    )


def print_simulation_report(result: SimulationBatchResult) -> None:
    """Print a formatted simulation report."""
    stats = result.stats
    
    print("\n" + "=" * 60)
    print("SPREAD SIMULATION RESULTS")
    print("=" * 60)
    
    print(f"\nSpread: {result.definition}")
    print(f"Initial Credit: ${result.initial_credit:.2f}")
    print(f"Max Profit: ${stats['max_profit']:.2f}")
    print(f"Max Loss: ${stats['max_loss']:.2f}")
    print(f"Paths Simulated: {result.n_paths:,}")
    
    print("\nP&L Distribution:")
    print(f"  Mean:           ${stats['mean_pnl']:.2f}")
    print(f"  Std Dev:        ${stats['std_pnl']:.2f}")
    print(f"  Median:         ${stats['median_pnl']:.2f}")
    print(f"  Min:            ${stats['min_pnl']:.2f}")
    print(f"  Max:            ${stats['max_pnl']:.2f}")
    
    print("\nRisk Metrics:")
    print(f"  Prob of Profit: {stats['pop']:.1%}")
    print(f"  Expected Value: ${stats['expected_value']:.2f}")
    print(f"  VaR (95%):      ${stats['var_95']:.2f}")
    print(f"  CVaR (95%):     ${stats['cvar_95']:.2f}")
    
    print("\nExit Reason Distribution:")
    for reason, pct in stats['exit_reason_pcts'].items():
        if pct > 0:
            print(f"  {reason:<20} {pct:.1%}")
    
    print(f"\nAverage Days Held: {stats['avg_days_held']:.1f}")
    
    print("=" * 60)
