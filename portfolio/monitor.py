"""
Position Monitor

This module monitors open positions against current market data,
calculates mark-to-market P&L, and checks exit conditions.
"""

from datetime import datetime, date
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from enum import Enum

from .positions import SpreadPosition, PositionStore, print_position_summary
from .market_data import (
    MarketDataStore, MarketSnapshot, OptionPriceUpdate,
    print_market_summary
)


class ExitSignal(Enum):
    """Exit signal types."""
    NONE = "none"
    PROFIT_TARGET = "profit_target"
    LOSS_LIMIT = "loss_limit"
    DTE_THRESHOLD = "dte_threshold"
    EXPIRING_TODAY = "expiring_today"
    MANUAL = "manual"


@dataclass
class PositionMark:
    """
    Mark-to-market valuation of a position.
    """
    position: SpreadPosition
    
    # Current market data
    current_underlying: float
    current_short_bid: float
    current_short_ask: float
    current_long_bid: float
    current_long_ask: float
    current_short_iv: Optional[float] = None
    
    # Calculated values
    spread_value_bid: float = 0.0     # Cost to close at bid (best case)
    spread_value_ask: float = 0.0     # Cost to close at ask (worst case)
    spread_value_mid: float = 0.0     # Mid mark
    
    unrealized_pnl_bid: float = 0.0   # P&L if closed at bid
    unrealized_pnl_mid: float = 0.0   # P&L at mid
    unrealized_pnl_ask: float = 0.0   # P&L if closed at ask
    
    pnl_pct_of_max: float = 0.0       # P&L as % of max profit
    
    # Exit signals
    exit_signal: ExitSignal = ExitSignal.NONE
    exit_message: str = ""
    
    # Timestamp
    mark_timestamp: str = ""
    
    def __post_init__(self):
        # Calculate spread values
        # To close: buy back short (pay ask), sell long (receive bid)
        self.spread_value_ask = self.current_short_ask - self.current_long_bid
        self.spread_value_bid = self.current_short_bid - self.current_long_ask
        self.spread_value_mid = (self.spread_value_bid + self.spread_value_ask) / 2
        
        # Calculate P&L (credit received - cost to close)
        credit = self.position.entry_credit
        self.unrealized_pnl_bid = credit - self.spread_value_bid
        self.unrealized_pnl_mid = credit - self.spread_value_mid
        self.unrealized_pnl_ask = credit - self.spread_value_ask
        
        # P&L as percentage of max profit
        if self.position.max_profit > 0:
            self.pnl_pct_of_max = self.unrealized_pnl_mid / self.position.max_profit
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L (mid) for all contracts."""
        return self.unrealized_pnl_mid * self.position.contracts * 100


@dataclass
class ExitRuleConfig:
    """Configuration for exit rules."""
    
    profit_target_pct: float = 0.50      # Close at 50% of max profit
    loss_multiple: float = 2.0            # Close when spread value = 2x credit
    dte_threshold: int = 7                # Close 7 days before expiration
    
    use_profit_target: bool = True
    use_loss_limit: bool = True
    use_dte_threshold: bool = True


class PositionMonitor:
    """
    Monitors positions and generates exit signals.
    
    Usage:
        monitor = PositionMonitor(
            position_store=PositionStore("positions.json"),
            market_store=MarketDataStore("market_data.json")
        )
        
        # Update market data
        monitor.market_store.add_market_snapshot(...)
        monitor.market_store.add_option_price(...)
        
        # Check positions
        marks = monitor.mark_all_positions()
        alerts = monitor.check_exit_signals()
    """
    
    def __init__(
        self,
        position_store: PositionStore,
        market_store: MarketDataStore,
        exit_rules: Optional[ExitRuleConfig] = None
    ):
        self.positions = position_store
        self.market = market_store
        self.exit_rules = exit_rules or ExitRuleConfig()
    
    def mark_position(self, position: SpreadPosition) -> Optional[PositionMark]:
        """
        Mark a single position to market.
        
        Returns None if market data is not available.
        """
        if position.status != "open":
            return None
        
        # Get option prices for both legs
        short_price = self.market.get_option_price(
            position.option_type,
            position.expiration_date,
            position.short_strike
        )
        long_price = self.market.get_option_price(
            position.option_type,
            position.expiration_date,
            position.long_strike
        )
        
        # Get underlying price
        snapshot = self.market.get_latest_snapshot()
        
        if short_price is None or long_price is None:
            return None
        
        underlying = snapshot.tlt_price if snapshot else position.entry_underlying
        
        mark = PositionMark(
            position=position,
            current_underlying=underlying,
            current_short_bid=short_price.bid,
            current_short_ask=short_price.ask,
            current_long_bid=long_price.bid,
            current_long_ask=long_price.ask,
            current_short_iv=short_price.implied_vol,
            mark_timestamp=datetime.now().isoformat()
        )
        
        # Check exit signals
        mark.exit_signal, mark.exit_message = self._check_exit_conditions(mark)
        
        return mark
    
    def _check_exit_conditions(
        self,
        mark: PositionMark
    ) -> Tuple[ExitSignal, str]:
        """Check all exit conditions for a position."""
        pos = mark.position
        rules = self.exit_rules
        
        # Check expiring today
        if pos.dte <= 0:
            return ExitSignal.EXPIRING_TODAY, "Position expiring today!"
        
        # Check DTE threshold
        if rules.use_dte_threshold and pos.dte <= rules.dte_threshold:
            return (
                ExitSignal.DTE_THRESHOLD,
                f"DTE ({pos.dte}) at or below threshold ({rules.dte_threshold})"
            )
        
        # Check profit target
        if rules.use_profit_target:
            if mark.pnl_pct_of_max >= rules.profit_target_pct:
                return (
                    ExitSignal.PROFIT_TARGET,
                    f"P&L ({mark.pnl_pct_of_max:.0%}) reached target ({rules.profit_target_pct:.0%})"
                )
        
        # Check loss limit
        if rules.use_loss_limit:
            loss_threshold = pos.entry_credit * rules.loss_multiple
            if mark.spread_value_mid >= loss_threshold:
                return (
                    ExitSignal.LOSS_LIMIT,
                    f"Spread value (${mark.spread_value_mid:.2f}) exceeds "
                    f"{rules.loss_multiple:.1f}x credit (${loss_threshold:.2f})"
                )
        
        return ExitSignal.NONE, ""
    
    def mark_all_positions(self) -> List[PositionMark]:
        """Mark all open positions to market."""
        marks = []
        
        for position in self.positions.get_open_positions():
            mark = self.mark_position(position)
            if mark:
                marks.append(mark)
        
        return marks
    
    def get_exit_signals(self) -> List[PositionMark]:
        """Get positions with active exit signals."""
        marks = self.mark_all_positions()
        return [m for m in marks if m.exit_signal != ExitSignal.NONE]
    
    def get_positions_missing_data(self) -> List[SpreadPosition]:
        """Get open positions without current market data."""
        missing = []
        
        for position in self.positions.get_open_positions():
            mark = self.mark_position(position)
            if mark is None:
                missing.append(position)
        
        return missing
    
    def close_position_from_mark(
        self,
        mark: PositionMark,
        use_mid: bool = True,
        exit_reason: Optional[str] = None
    ):
        """
        Close a position using current market data.
        
        Parameters
        ----------
        mark : PositionMark
            Position mark with current prices
        use_mid : bool
            Use mid price (True) or ask price (False, conservative)
        exit_reason : str, optional
            Override exit reason
        """
        exit_debit = mark.spread_value_mid if use_mid else mark.spread_value_ask
        reason = exit_reason or mark.exit_signal.value
        
        self.positions.close_position(
            position_id=mark.position.position_id,
            exit_debit=exit_debit,
            exit_reason=reason,
            exit_underlying=mark.current_underlying
        )


def print_position_marks(marks: List[PositionMark], title: str = "Position Marks"):
    """Print position marks table."""
    print("\n" + "=" * 120)
    print(title.upper())
    print("=" * 120)
    
    if not marks:
        print("\nNo positions to display.")
        return
    
    print(f"\n{'ID':<10} {'Type':<6} {'Strikes':<12} {'Exp':<12} {'DTE':<6} "
          f"{'Entry':<10} {'Current':<10} {'P&L':<12} {'%Max':<8} {'Signal':<15}")
    print("-" * 120)
    
    total_pnl = 0
    
    for m in sorted(marks, key=lambda x: x.position.expiration_date):
        pos = m.position
        type_str = "Put" if pos.option_type == 'put' else "Call"
        strikes = f"{pos.short_strike}/{pos.long_strike}"
        
        # Color coding would be nice but keeping it simple
        pnl_str = f"${m.unrealized_pnl_mid:+.2f}"
        pct_str = f"{m.pnl_pct_of_max:+.0%}"
        signal_str = m.exit_signal.value if m.exit_signal != ExitSignal.NONE else ""
        
        print(f"{pos.position_id:<10} {type_str:<6} {strikes:<12} {pos.expiration_date:<12} "
              f"{pos.dte:<6} ${pos.entry_credit:<9.2f} ${m.spread_value_mid:<9.2f} "
              f"{pnl_str:<12} {pct_str:<8} {signal_str:<15}")
        
        total_pnl += m.total_unrealized_pnl
    
    print("-" * 120)
    print(f"{'TOTAL':<10} {'':<6} {'':<12} {'':<12} {'':<6} "
          f"{'':<10} {'':<10} ${total_pnl:+.2f}")
    
    # Show alerts
    alerts = [m for m in marks if m.exit_signal != ExitSignal.NONE]
    if alerts:
        print("\n⚠️  EXIT SIGNALS:")
        for m in alerts:
            print(f"  [{m.position.position_id}] {m.exit_message}")
    
    print("=" * 120)


def print_portfolio_summary(
    position_store: PositionStore,
    market_store: MarketDataStore,
    marks: List[PositionMark]
):
    """Print comprehensive portfolio summary."""
    print("\n" + "=" * 80)
    print("PORTFOLIO SUMMARY")
    print("=" * 80)
    
    # Latest market data
    snapshot = market_store.get_latest_snapshot()
    if snapshot:
        print(f"\nMarket Data ({snapshot.timestamp[:16]}):")
        print(f"  TLT: ${snapshot.tlt_price:.2f}")
        print(f"  20Y: {snapshot.yield_20y:.3%}  |  30Y: {snapshot.yield_30y:.3%}")
    
    # Open positions summary
    open_positions = position_store.get_open_positions()
    closed_positions = position_store.get_closed_positions()
    
    print(f"\nPositions:")
    print(f"  Open: {len(open_positions)}")
    print(f"  Closed: {len(closed_positions)}")
    
    if marks:
        # Calculate totals
        total_credit = sum(m.position.total_credit for m in marks)
        total_max_loss = sum(m.position.total_max_loss for m in marks)
        total_unrealized = sum(m.total_unrealized_pnl for m in marks)
        
        print(f"\nOpen Position Totals:")
        print(f"  Total Credit Received: ${total_credit:,.2f}")
        print(f"  Total Max Loss: ${total_max_loss:,.2f}")
        print(f"  Unrealized P&L: ${total_unrealized:+,.2f}")
        
        # Alerts
        alerts = [m for m in marks if m.exit_signal != ExitSignal.NONE]
        if alerts:
            print(f"\n⚠️  {len(alerts)} position(s) with exit signals!")
    
    # Realized P&L from closed positions
    if closed_positions:
        realized_pnl = sum(p.realized_pnl or 0 for p in closed_positions)
        print(f"\nClosed Position Totals:")
        print(f"  Realized P&L: ${realized_pnl:+,.2f}")
        print(f"  Trades: {len(closed_positions)}")
    
    print("=" * 80)
