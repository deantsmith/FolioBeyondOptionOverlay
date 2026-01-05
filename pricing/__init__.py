"""Option pricing and spread valuation modules."""

from .black_scholes import (
    OptionQuote,
    call_price, put_price,
    call_delta, put_delta,
    gamma, vega,
    call_theta, put_theta,
    price_call, price_put,
    find_strike_by_delta,
    implied_volatility,
    call_price_vec, put_price_vec
)

from .spreads import (
    SpreadType,
    SpreadDefinition,
    SpreadQuote,
    create_spread_by_delta,
    create_spread_by_strikes,
    price_spread,
    value_spread,
    value_spread_at_expiration,
    value_spread_vec,
    generate_spread_candidates,
    print_spread_summary
)

from .exits import (
    ExitReason,
    ExitRules,
    ExitResult,
    check_exit_conditions,
    simulate_position,
    simulate_position_fast,
    SimulationBatchResult,
    simulate_spread_batch,
    print_simulation_report
)

__all__ = [
    # Black-Scholes
    'OptionQuote',
    'call_price', 'put_price',
    'call_delta', 'put_delta',
    'gamma', 'vega',
    'call_theta', 'put_theta',
    'price_call', 'price_put',
    'find_strike_by_delta',
    'implied_volatility',
    'call_price_vec', 'put_price_vec',
    
    # Spreads
    'SpreadType',
    'SpreadDefinition',
    'SpreadQuote',
    'create_spread_by_delta',
    'create_spread_by_strikes',
    'price_spread',
    'value_spread',
    'value_spread_at_expiration',
    'value_spread_vec',
    'generate_spread_candidates',
    'print_spread_summary',
    
    # Exits
    'ExitReason',
    'ExitRules',
    'ExitResult',
    'check_exit_conditions',
    'simulate_position',
    'simulate_position_fast',
    'SimulationBatchResult',
    'simulate_spread_batch',
    'print_simulation_report'
]
