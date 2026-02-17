"""
Shared evaluation helpers for strategy recommendation pipelines.

Centralizes risk-adjusted return calculation, constraint checks, and ranking
to keep strategy and market evaluation entry points aligned.
"""

from typing import Callable, Iterable, List, Optional, Tuple, TypeVar

from pricing.exits import SimulationBatchResult

T = TypeVar("T")


def compute_risk_adjusted_return(expected_return: float, cvar_95: float) -> float:
    """Compute risk-adjusted return with a consistent CVaR guardrail."""
    if abs(cvar_95) > 0.01:
        return expected_return / abs(cvar_95)
    return float("inf") if expected_return > 0 else 0.0


def extract_core_metrics(
    batch_result: SimulationBatchResult,
) -> Tuple[float, float, float, float]:
    """
    Extract expected return, POP, CVaR (95%), and risk-adjusted return.
    """
    stats = batch_result.stats
    expected_return = stats["expected_value"]
    pop = stats["pop"]
    cvar_95 = stats["cvar_95"]
    risk_adjusted_return = compute_risk_adjusted_return(expected_return, cvar_95)
    return expected_return, pop, cvar_95, risk_adjusted_return


def apply_constraints(
    pop: float,
    cvar_95: float,
    min_pop: float,
    max_cvar_loss: Optional[float]
) -> Tuple[bool, List[str]]:
    """Apply POP and optional CVaR loss constraints."""
    violations: List[str] = []

    if pop < min_pop:
        violations.append(f"POP {pop:.1%} < min {min_pop:.1%}")

    if max_cvar_loss is not None:
        if cvar_95 < -max_cvar_loss:
            violations.append(
                f"CVaR ${cvar_95:.2f} exceeds budget ${-max_cvar_loss:.2f}"
            )

    return len(violations) == 0, violations


def rank_results(
    items: Iterable[T],
    get_rar: Callable[[T], float],
    get_passes: Callable[[T], bool],
    set_rank: Callable[[T, int], None]
) -> List[T]:
    """
    Rank items by risk-adjusted return (descending), and assign ranks.
    Items failing constraints are ranked after all passing items.
    """
    items_list = list(items)
    passing = [item for item in items_list if get_passes(item)]
    failing = [item for item in items_list if not get_passes(item)]

    passing.sort(key=get_rar, reverse=True)

    for i, item in enumerate(passing):
        set_rank(item, i + 1)

    for item in failing:
        set_rank(item, len(items_list))

    return passing + failing
