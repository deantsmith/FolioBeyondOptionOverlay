"""Strategy evaluation and risk metrics modules."""

from .evaluation import (
    CalibratedModel,
    StrategyCandidate,
    EvaluationConfig,
    StrategyEvaluator,
    run_evaluation
)

from .metrics import (
    probability_of_profit,
    expected_return,
    value_at_risk,
    conditional_var,
    sortino_ratio,
    calmar_ratio,
    win_loss_ratio,
    profit_factor,
    expectancy,
    kelly_criterion,
    RiskMetrics,
    compute_all_metrics,
    print_metrics_report,
    compare_strategies
)

__all__ = [
    # Evaluation
    'CalibratedModel',
    'StrategyCandidate',
    'EvaluationConfig',
    'StrategyEvaluator',
    'run_evaluation',
    
    # Metrics
    'probability_of_profit',
    'expected_return',
    'value_at_risk',
    'conditional_var',
    'sortino_ratio',
    'calmar_ratio',
    'win_loss_ratio',
    'profit_factor',
    'expectancy',
    'kelly_criterion',
    'RiskMetrics',
    'compute_all_metrics',
    'print_metrics_report',
    'compare_strategies'
]
