"""
Risk Metrics

This module provides additional risk metrics and analysis functions
for evaluating option strategies.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


def probability_of_profit(pnls: np.ndarray) -> float:
    """Calculate probability of profit."""
    return np.mean(pnls > 0)


def expected_return(pnls: np.ndarray) -> float:
    """Calculate expected (mean) return."""
    return np.mean(pnls)


def value_at_risk(pnls: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk.
    
    Parameters
    ----------
    pnls : np.ndarray
        P&L distribution.
    confidence : float
        Confidence level (e.g., 0.95 for 95% VaR).
        
    Returns
    -------
    float
        VaR (negative number representing potential loss).
    """
    return np.percentile(pnls, (1 - confidence) * 100)


def conditional_var(pnls: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).
    
    Parameters
    ----------
    pnls : np.ndarray
        P&L distribution.
    confidence : float
        Confidence level.
        
    Returns
    -------
    float
        CVaR (average of losses beyond VaR).
    """
    var = value_at_risk(pnls, confidence)
    return np.mean(pnls[pnls <= var])


def sortino_ratio(
    pnls: np.ndarray,
    target_return: float = 0.0,
    annualization: float = 12.0  # Monthly strategies
) -> float:
    """
    Calculate Sortino Ratio.
    
    Parameters
    ----------
    pnls : np.ndarray
        P&L distribution.
    target_return : float
        Minimum acceptable return.
    annualization : float
        Factor to annualize (12 for monthly).
        
    Returns
    -------
    float
        Sortino ratio.
    """
    excess_returns = pnls - target_return
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')
    
    downside_std = np.std(downside_returns)
    
    if downside_std < 1e-10:
        return float('inf') if np.mean(excess_returns) > 0 else 0
    
    mean_excess = np.mean(excess_returns)
    
    return (mean_excess / downside_std) * np.sqrt(annualization)


def calmar_ratio(pnls: np.ndarray) -> float:
    """
    Calculate Calmar-like ratio (return / max drawdown).
    
    For single-period trades, uses mean return / max loss.
    """
    max_loss = abs(np.min(pnls))
    
    if max_loss < 1e-10:
        return float('inf') if np.mean(pnls) > 0 else 0
    
    return np.mean(pnls) / max_loss


def win_loss_ratio(pnls: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate win/loss statistics.
    
    Returns
    -------
    Tuple[float, float, float]
        (average_win, average_loss, win_loss_ratio)
    """
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    
    if abs(avg_loss) < 1e-10:
        ratio = float('inf') if avg_win > 0 else 0
    else:
        ratio = avg_win / abs(avg_loss)
    
    return avg_win, avg_loss, ratio


def profit_factor(pnls: np.ndarray) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    """
    gross_profit = np.sum(pnls[pnls > 0])
    gross_loss = abs(np.sum(pnls[pnls < 0]))
    
    if gross_loss < 1e-10:
        return float('inf') if gross_profit > 0 else 0
    
    return gross_profit / gross_loss


def expectancy(pnls: np.ndarray) -> float:
    """
    Calculate expectancy (expected value per trade).
    
    E = (P(win) * avg_win) + (P(loss) * avg_loss)
    """
    p_win = np.mean(pnls > 0)
    p_loss = np.mean(pnls < 0)
    
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    
    return (p_win * avg_win) + (p_loss * avg_loss)


def kelly_criterion(pnls: np.ndarray, max_fraction: float = 0.25) -> float:
    """
    Calculate Kelly criterion for position sizing.
    
    Parameters
    ----------
    pnls : np.ndarray
        P&L distribution.
    max_fraction : float
        Maximum fraction to risk.
        
    Returns
    -------
    float
        Recommended fraction of capital to risk.
    """
    p_win = np.mean(pnls > 0)
    
    avg_win, avg_loss, wl_ratio = win_loss_ratio(pnls)
    
    if wl_ratio == 0 or avg_loss >= 0:
        return 0
    
    # Kelly formula: f = p - q/b where b = avg_win/|avg_loss|
    b = wl_ratio
    f = p_win - (1 - p_win) / b
    
    # Cap at max_fraction and ensure non-negative
    return max(0, min(f, max_fraction))


@dataclass
class RiskMetrics:
    """Container for comprehensive risk metrics."""
    
    # Basic statistics
    mean_pnl: float
    std_pnl: float
    median_pnl: float
    min_pnl: float
    max_pnl: float
    
    # Probabilities
    pop: float              # Probability of profit
    
    # Risk measures
    var_95: float          # 95% Value at Risk
    var_99: float          # 99% Value at Risk
    cvar_95: float         # 95% Conditional VaR
    cvar_99: float         # 99% Conditional VaR
    
    # Ratios
    risk_adjusted_return: float  # E[R] / |CVaR_95|
    sortino_ratio: float
    profit_factor: float
    win_loss_ratio: float
    
    # Other
    expectancy: float
    kelly_fraction: float
    
    def __repr__(self):
        return (f"RiskMetrics(POP={self.pop:.1%}, E[R]=${self.mean_pnl:.2f}, "
                f"CVaR95=${self.cvar_95:.2f}, RAR={self.risk_adjusted_return:.2f})")


def compute_all_metrics(pnls: np.ndarray) -> RiskMetrics:
    """
    Compute comprehensive risk metrics.
    
    Parameters
    ----------
    pnls : np.ndarray
        P&L distribution.
        
    Returns
    -------
    RiskMetrics
        Complete risk metrics.
    """
    avg_win, avg_loss, wl_ratio = win_loss_ratio(pnls)
    cvar_95 = conditional_var(pnls, 0.95)
    
    # Risk-adjusted return
    if abs(cvar_95) > 0.01:
        rar = np.mean(pnls) / abs(cvar_95)
    else:
        rar = float('inf') if np.mean(pnls) > 0 else 0
    
    return RiskMetrics(
        mean_pnl=np.mean(pnls),
        std_pnl=np.std(pnls),
        median_pnl=np.median(pnls),
        min_pnl=np.min(pnls),
        max_pnl=np.max(pnls),
        pop=probability_of_profit(pnls),
        var_95=value_at_risk(pnls, 0.95),
        var_99=value_at_risk(pnls, 0.99),
        cvar_95=cvar_95,
        cvar_99=conditional_var(pnls, 0.99),
        risk_adjusted_return=rar,
        sortino_ratio=sortino_ratio(pnls),
        profit_factor=profit_factor(pnls),
        win_loss_ratio=wl_ratio,
        expectancy=expectancy(pnls),
        kelly_fraction=kelly_criterion(pnls)
    )


def print_metrics_report(metrics: RiskMetrics, title: str = "Risk Metrics") -> None:
    """Print formatted risk metrics report."""
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)
    
    print("\nP&L Statistics:")
    print(f"  Mean:               ${metrics.mean_pnl:.2f}")
    print(f"  Std Dev:            ${metrics.std_pnl:.2f}")
    print(f"  Median:             ${metrics.median_pnl:.2f}")
    print(f"  Min:                ${metrics.min_pnl:.2f}")
    print(f"  Max:                ${metrics.max_pnl:.2f}")
    
    print("\nProbabilities:")
    print(f"  Prob of Profit:     {metrics.pop:.1%}")
    
    print("\nRisk Measures:")
    print(f"  VaR (95%):          ${metrics.var_95:.2f}")
    print(f"  VaR (99%):          ${metrics.var_99:.2f}")
    print(f"  CVaR (95%):         ${metrics.cvar_95:.2f}")
    print(f"  CVaR (99%):         ${metrics.cvar_99:.2f}")
    
    print("\nRatios:")
    print(f"  Risk-Adj Return:    {metrics.risk_adjusted_return:.2f}")
    print(f"  Sortino Ratio:      {metrics.sortino_ratio:.2f}")
    print(f"  Profit Factor:      {metrics.profit_factor:.2f}")
    print(f"  Win/Loss Ratio:     {metrics.win_loss_ratio:.2f}")
    
    print("\nPosition Sizing:")
    print(f"  Expectancy:         ${metrics.expectancy:.2f}")
    print(f"  Kelly Fraction:     {metrics.kelly_fraction:.1%}")
    
    print("=" * 60)


def compare_strategies(
    strategy_results: Dict[str, np.ndarray],
    names: Optional[List[str]] = None
) -> None:
    """
    Compare multiple strategies side by side.
    
    Parameters
    ----------
    strategy_results : Dict[str, np.ndarray]
        Dictionary mapping strategy names to P&L arrays.
    names : List[str], optional
        Order of strategies to display.
    """
    if names is None:
        names = list(strategy_results.keys())
    
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON")
    print("=" * 100)
    
    # Header
    header = f"{'Metric':<20}"
    for name in names:
        header += f"{name:<15}"
    print(header)
    print("-" * 100)
    
    # Compute metrics for each
    all_metrics = {name: compute_all_metrics(strategy_results[name]) for name in names}
    
    # Rows
    metrics_to_show = [
        ('POP', 'pop', lambda x: f"{x:.1%}"),
        ('Mean P&L', 'mean_pnl', lambda x: f"${x:.2f}"),
        ('Std Dev', 'std_pnl', lambda x: f"${x:.2f}"),
        ('CVaR (95%)', 'cvar_95', lambda x: f"${x:.2f}"),
        ('Risk-Adj Ret', 'risk_adjusted_return', lambda x: f"{x:.2f}"),
        ('Profit Factor', 'profit_factor', lambda x: f"{x:.2f}"),
        ('Win/Loss', 'win_loss_ratio', lambda x: f"{x:.2f}"),
        ('Kelly %', 'kelly_fraction', lambda x: f"{x:.1%}"),
    ]
    
    for label, attr, fmt in metrics_to_show:
        row = f"{label:<20}"
        for name in names:
            value = getattr(all_metrics[name], attr)
            row += f"{fmt(value):<15}"
        print(row)
    
    print("=" * 100)
