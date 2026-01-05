"""
Credit Spread Valuation

This module handles pricing and P&L calculation for vertical credit spreads
(bull put spreads and bear call spreads) on TLT.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum

from .black_scholes import (
    call_price, put_price,
    call_price_vec, put_price_vec,
    call_delta, put_delta,
    find_strike_by_delta,
    price_call, price_put,
    OptionQuote
)


class SpreadType(Enum):
    """Type of credit spread."""
    BULL_PUT = "bull_put"      # Sell put, buy lower put
    BEAR_CALL = "bear_call"    # Sell call, buy higher call


@dataclass
class SpreadDefinition:
    """Definition of a credit spread."""
    spread_type: SpreadType
    short_strike: float
    long_strike: float
    expiration_days: int
    
    # Optional: how the spread was defined
    short_delta: Optional[float] = None
    spread_width: Optional[float] = None
    
    @property
    def width(self) -> float:
        """Width of the spread in dollars."""
        return abs(self.long_strike - self.short_strike)
    
    @property
    def is_put_spread(self) -> bool:
        return self.spread_type == SpreadType.BULL_PUT
    
    @property
    def is_call_spread(self) -> bool:
        return self.spread_type == SpreadType.BEAR_CALL
    
    def __repr__(self):
        if self.is_put_spread:
            return (f"BullPutSpread({self.short_strike}/{self.long_strike}, "
                    f"{self.expiration_days}DTE)")
        else:
            return (f"BearCallSpread({self.short_strike}/{self.long_strike}, "
                    f"{self.expiration_days}DTE)")


@dataclass
class SpreadQuote:
    """Pricing quote for a credit spread."""
    definition: SpreadDefinition
    
    # Prices
    short_price: float
    long_price: float
    net_credit: float
    
    # Greeks (net)
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    
    # Risk metrics
    max_profit: float      # = net credit
    max_loss: float        # = spread width - net credit
    breakeven: float       # Strike where P&L = 0
    
    # Current value (for marking to market)
    current_value: float = 0.0  # Cost to close position
    unrealized_pnl: float = 0.0
    
    def __repr__(self):
        return (f"SpreadQuote(credit=${self.net_credit:.2f}, "
                f"max_loss=${self.max_loss:.2f}, "
                f"Δ={self.net_delta:.3f})")


def create_spread_by_delta(
    spot: float,
    r: float,
    sigma: float,
    expiration_days: int,
    target_delta: float,
    spread_width: float,
    spread_type: SpreadType
) -> SpreadDefinition:
    """
    Create a spread definition by targeting a specific delta for the short strike.
    
    Parameters
    ----------
    spot : float
        Current TLT price.
    r : float
        Risk-free rate.
    sigma : float
        Implied volatility.
    expiration_days : int
        Days to expiration.
    target_delta : float
        Target delta for short strike (absolute value, e.g., 0.20).
    spread_width : float
        Width of spread in dollars.
    spread_type : SpreadType
        BULL_PUT or BEAR_CALL.
        
    Returns
    -------
    SpreadDefinition
        Spread with strikes determined by delta targeting.
    """
    T = expiration_days / 252.0
    
    if spread_type == SpreadType.BULL_PUT:
        # Find put strike with target delta
        short_strike = find_strike_by_delta(
            spot, r, sigma, T, target_delta, 'put'
        )
        # Round to nearest dollar
        short_strike = round(short_strike)
        # Long strike is below short strike
        long_strike = short_strike - spread_width
    else:
        # Find call strike with target delta
        short_strike = find_strike_by_delta(
            spot, r, sigma, T, target_delta, 'call'
        )
        short_strike = round(short_strike)
        # Long strike is above short strike
        long_strike = short_strike + spread_width
    
    return SpreadDefinition(
        spread_type=spread_type,
        short_strike=short_strike,
        long_strike=long_strike,
        expiration_days=expiration_days,
        short_delta=target_delta,
        spread_width=spread_width
    )


def create_spread_by_strikes(
    short_strike: float,
    long_strike: float,
    expiration_days: int,
    spread_type: SpreadType
) -> SpreadDefinition:
    """
    Create a spread definition with explicit strikes.
    
    Parameters
    ----------
    short_strike : float
        Strike of the short option.
    long_strike : float
        Strike of the long option.
    expiration_days : int
        Days to expiration.
    spread_type : SpreadType
        BULL_PUT or BEAR_CALL.
        
    Returns
    -------
    SpreadDefinition
        Spread definition.
    """
    return SpreadDefinition(
        spread_type=spread_type,
        short_strike=short_strike,
        long_strike=long_strike,
        expiration_days=expiration_days,
        spread_width=abs(short_strike - long_strike)
    )


def price_spread(
    definition: SpreadDefinition,
    spot: float,
    r: float,
    sigma: float,
    days_remaining: Optional[int] = None
) -> SpreadQuote:
    """
    Price a credit spread.
    
    Parameters
    ----------
    definition : SpreadDefinition
        Spread to price.
    spot : float
        Current underlying price.
    r : float
        Risk-free rate.
    sigma : float
        Implied volatility.
    days_remaining : int, optional
        Days until expiration. If None, uses definition's expiration_days.
        
    Returns
    -------
    SpreadQuote
        Complete pricing quote.
    """
    if days_remaining is None:
        days_remaining = definition.expiration_days
    
    T = days_remaining / 252.0
    
    if definition.is_put_spread:
        short_quote = price_put(spot, definition.short_strike, r, sigma, T)
        long_quote = price_put(spot, definition.long_strike, r, sigma, T)
    else:
        short_quote = price_call(spot, definition.short_strike, r, sigma, T)
        long_quote = price_call(spot, definition.long_strike, r, sigma, T)
    
    # Net credit = what we receive (short) - what we pay (long)
    net_credit = short_quote.price - long_quote.price
    
    # Net Greeks (short - long, since we're short the more expensive option)
    net_delta = -short_quote.delta + long_quote.delta  # Negative of short position
    net_gamma = -short_quote.gamma + long_quote.gamma
    net_theta = -short_quote.theta + long_quote.theta  # Positive theta for credit spreads
    net_vega = -short_quote.vega + long_quote.vega
    
    # Risk metrics
    max_profit = net_credit
    max_loss = definition.width - net_credit
    
    if definition.is_put_spread:
        # Breakeven for bull put = short strike - net credit
        breakeven = definition.short_strike - net_credit
    else:
        # Breakeven for bear call = short strike + net credit
        breakeven = definition.short_strike + net_credit
    
    return SpreadQuote(
        definition=definition,
        short_price=short_quote.price,
        long_price=long_quote.price,
        net_credit=net_credit,
        net_delta=net_delta,
        net_gamma=net_gamma,
        net_theta=net_theta,
        net_vega=net_vega,
        max_profit=max_profit,
        max_loss=max_loss,
        breakeven=breakeven,
        current_value=net_credit,  # At open, value = credit received
        unrealized_pnl=0.0
    )


def value_spread(
    definition: SpreadDefinition,
    spot: float,
    r: float,
    sigma: float,
    days_remaining: int,
    initial_credit: float
) -> Tuple[float, float]:
    """
    Calculate current value and P&L of an existing spread position.
    
    Parameters
    ----------
    definition : SpreadDefinition
        Spread definition.
    spot : float
        Current underlying price.
    r : float
        Risk-free rate.
    sigma : float
        Current implied volatility.
    days_remaining : int
        Days until expiration.
    initial_credit : float
        Credit received when opening position.
        
    Returns
    -------
    Tuple[float, float]
        (cost_to_close, unrealized_pnl)
        Positive P&L = profit, negative = loss
    """
    T = days_remaining / 252.0
    
    if definition.is_put_spread:
        short_value = put_price(spot, definition.short_strike, r, sigma, T)
        long_value = put_price(spot, definition.long_strike, r, sigma, T)
    else:
        short_value = call_price(spot, definition.short_strike, r, sigma, T)
        long_value = call_price(spot, definition.long_strike, r, sigma, T)
    
    # Cost to close = buy back short - sell long
    cost_to_close = short_value - long_value
    
    # P&L = credit received - cost to close
    unrealized_pnl = initial_credit - cost_to_close
    
    return cost_to_close, unrealized_pnl


def value_spread_at_expiration(
    definition: SpreadDefinition,
    spot: float,
    initial_credit: float
) -> float:
    """
    Calculate P&L at expiration.
    
    Parameters
    ----------
    definition : SpreadDefinition
        Spread definition.
    spot : float
        Underlying price at expiration.
    initial_credit : float
        Credit received when opening.
        
    Returns
    -------
    float
        P&L at expiration (positive = profit).
    """
    if definition.is_put_spread:
        # Bull put spread
        if spot >= definition.short_strike:
            # Both expire worthless - keep full credit
            return initial_credit
        elif spot <= definition.long_strike:
            # Max loss
            return initial_credit - definition.width
        else:
            # Partial loss: short put is ITM
            intrinsic = definition.short_strike - spot
            return initial_credit - intrinsic
    else:
        # Bear call spread
        if spot <= definition.short_strike:
            # Both expire worthless - keep full credit
            return initial_credit
        elif spot >= definition.long_strike:
            # Max loss
            return initial_credit - definition.width
        else:
            # Partial loss: short call is ITM
            intrinsic = spot - definition.short_strike
            return initial_credit - intrinsic


def value_spread_vec(
    definition: SpreadDefinition,
    spots: np.ndarray,
    r: float,
    sigmas: np.ndarray,
    days_remaining: int,
    initial_credit: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized spread valuation for simulation paths.
    
    Parameters
    ----------
    definition : SpreadDefinition
        Spread definition.
    spots : np.ndarray
        Array of underlying prices.
    r : float
        Risk-free rate.
    sigmas : np.ndarray
        Array of implied volatilities.
    days_remaining : int
        Days until expiration.
    initial_credit : float
        Credit received when opening.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (costs_to_close, unrealized_pnls)
    """
    T = days_remaining / 252.0
    
    if T <= 0:
        # At expiration, use intrinsic values
        pnls = np.array([
            value_spread_at_expiration(definition, s, initial_credit)
            for s in spots
        ])
        costs = initial_credit - pnls
        return costs, pnls
    
    if definition.is_put_spread:
        short_values = put_price_vec(spots, definition.short_strike, r, sigmas, T)
        long_values = put_price_vec(spots, definition.long_strike, r, sigmas, T)
    else:
        short_values = call_price_vec(spots, definition.short_strike, r, sigmas, T)
        long_values = call_price_vec(spots, definition.long_strike, r, sigmas, T)
    
    costs_to_close = short_values - long_values
    unrealized_pnls = initial_credit - costs_to_close
    
    return costs_to_close, unrealized_pnls


def generate_spread_candidates(
    spot: float,
    r: float,
    sigma: float,
    expiration_days: int,
    delta_targets: List[float] = [0.15, 0.20, 0.25, 0.30],
    spread_widths: List[float] = [2, 3, 5],
    spread_types: List[SpreadType] = None
) -> List[SpreadQuote]:
    """
    Generate a set of candidate spreads for evaluation.
    
    Parameters
    ----------
    spot : float
        Current underlying price.
    r : float
        Risk-free rate.
    sigma : float
        Implied volatility.
    expiration_days : int
        Days to expiration.
    delta_targets : List[float]
        Target deltas for short strikes.
    spread_widths : List[float]
        Spread widths to consider.
    spread_types : List[SpreadType], optional
        Spread types to include. Default: both put and call spreads.
        
    Returns
    -------
    List[SpreadQuote]
        Priced spread candidates.
    """
    if spread_types is None:
        spread_types = [SpreadType.BULL_PUT, SpreadType.BEAR_CALL]
    
    candidates = []
    
    for spread_type in spread_types:
        for delta in delta_targets:
            for width in spread_widths:
                try:
                    definition = create_spread_by_delta(
                        spot, r, sigma, expiration_days,
                        delta, width, spread_type
                    )
                    quote = price_spread(definition, spot, r, sigma)
                    candidates.append(quote)
                except Exception as e:
                    # Skip invalid combinations
                    continue
    
    return candidates


def print_spread_summary(quotes: List[SpreadQuote]) -> None:
    """Print a summary table of spread candidates."""
    print("\n" + "=" * 80)
    print("SPREAD CANDIDATES")
    print("=" * 80)
    
    print(f"\n{'Type':<12} {'Strikes':<12} {'Delta':<8} {'Credit':<10} "
          f"{'MaxLoss':<10} {'RR':<8} {'Theta':<8}")
    print("-" * 80)
    
    for q in quotes:
        spread_type = "Put" if q.definition.is_put_spread else "Call"
        strikes = f"{q.definition.short_strike}/{q.definition.long_strike}"
        delta = f"{abs(q.net_delta):.2f}"
        credit = f"${q.net_credit:.2f}"
        max_loss = f"${q.max_loss:.2f}"
        risk_reward = f"{q.max_profit/q.max_loss:.2f}" if q.max_loss > 0 else "N/A"
        theta = f"${q.net_theta:.3f}"
        
        print(f"{spread_type:<12} {strikes:<12} {delta:<8} {credit:<10} "
              f"{max_loss:<10} {risk_reward:<8} {theta:<8}")
    
    print("=" * 80)
