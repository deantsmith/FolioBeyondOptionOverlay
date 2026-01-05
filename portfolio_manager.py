#!/usr/bin/env python
"""
Portfolio Manager

Interactive command-line interface for managing TLT option spread positions.

Features:
- Enter new positions
- Update market data (yields, TLT, option prices)
- Mark positions to market
- Check exit signals
- Record position closes
- View portfolio summary and history

Usage:
    python portfolio_manager.py [--data-dir ./data]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, date

# Add script directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from portfolio import (
    SpreadPosition, PositionStore, create_position,
    print_position_summary, print_position_detail,
    MarketSnapshot, OptionPriceUpdate, MarketDataStore,
    print_market_summary, print_option_prices,
    ExitSignal, ExitRuleConfig, PositionMonitor,
    print_position_marks, print_portfolio_summary
)


class PortfolioManager:
    """Interactive portfolio manager."""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize stores
        self.positions = PositionStore(self.data_dir / "positions.json")
        self.market = MarketDataStore(self.data_dir / "market_data.json")
        
        # Initialize monitor with default exit rules
        self.exit_rules = ExitRuleConfig(
            profit_target_pct=0.50,
            loss_multiple=2.0,
            dte_threshold=7
        )
        self.monitor = PositionMonitor(self.positions, self.market, self.exit_rules)
    
    def run(self):
        """Run interactive menu."""
        while True:
            self._print_menu()
            choice = input("\nEnter choice: ").strip()
            
            if choice == '1':
                self._show_dashboard()
            elif choice == '2':
                self._add_position()
            elif choice == '3':
                self._update_market_snapshot()
            elif choice == '4':
                self._update_option_prices()
            elif choice == '5':
                self._mark_positions()
            elif choice == '6':
                self._close_position()
            elif choice == '7':
                self._view_positions()
            elif choice == '8':
                self._view_market_data()
            elif choice == '9':
                self._import_bloomberg()
            elif choice == '10':
                self._configure_exit_rules()
            elif choice == 'q' or choice == 'Q':
                print("\nExiting...")
                break
            else:
                print("\nInvalid choice. Please try again.")
    
    def _print_menu(self):
        """Print main menu."""
        print("\n" + "=" * 50)
        print("TLT OPTIONS PORTFOLIO MANAGER")
        print("=" * 50)
        print("\n  1. Dashboard (summary + alerts)")
        print("  2. Add new position")
        print("  3. Update market data (yields + TLT)")
        print("  4. Update option prices")
        print("  5. Mark positions to market")
        print("  6. Close position")
        print("  7. View all positions")
        print("  8. View market data")
        print("  9. Import Bloomberg CSV")
        print(" 10. Configure exit rules")
        print("  Q. Quit")
    
    def _show_dashboard(self):
        """Show portfolio dashboard with alerts."""
        marks = self.monitor.mark_all_positions()
        print_portfolio_summary(self.positions, self.market, marks)
        
        if marks:
            print_position_marks(marks, "Open Positions - Mark to Market")
        
        # Show positions missing data
        missing = self.monitor.get_positions_missing_data()
        if missing:
            print("\n⚠️  Positions missing market data:")
            for p in missing:
                print(f"    [{p.position_id}] {p.option_type.upper()} "
                      f"{p.short_strike}/{p.long_strike} exp {p.expiration_date}")
    
    def _add_position(self):
        """Add a new position."""
        print("\n--- ADD NEW POSITION ---\n")
        
        try:
            # Get position details
            option_type = input("Option type (put/call) [put]: ").strip().lower() or "put"
            if option_type not in ['put', 'call']:
                print("Invalid option type.")
                return
            
            short_strike = float(input("Short strike: "))
            long_strike = float(input("Long strike: "))
            expiration = input("Expiration date (YYYY-MM-DD): ").strip()
            
            # Validate expiration format
            datetime.strptime(expiration, "%Y-%m-%d")
            
            entry_credit = float(input("Entry credit per spread: $"))
            contracts = int(input("Number of contracts [1]: ") or "1")
            
            # Get current TLT price
            snapshot = self.market.get_latest_snapshot()
            if snapshot:
                default_underlying = snapshot.tlt_price
                underlying = input(f"TLT price at entry [{default_underlying:.2f}]: ").strip()
                underlying = float(underlying) if underlying else default_underlying
            else:
                underlying = float(input("TLT price at entry: "))
            
            # Optional fields
            iv = input("Short strike IV (%, optional): ").strip()
            entry_iv = float(iv) / 100 if iv else None
            
            delta = input("Short strike delta (optional): ").strip()
            entry_delta = float(delta) if delta else None
            
            entry_date = input(f"Entry date [{date.today().isoformat()}]: ").strip()
            entry_date = entry_date or date.today().isoformat()
            
            notes = input("Notes (optional): ").strip()
            
            # Create and save position
            position = create_position(
                option_type=option_type,
                short_strike=short_strike,
                long_strike=long_strike,
                expiration_date=expiration,
                entry_credit=entry_credit,
                contracts=contracts,
                entry_underlying=underlying,
                entry_short_iv=entry_iv,
                entry_delta=entry_delta,
                entry_date=entry_date,
                notes=notes
            )
            
            self.positions.add_position(position)
            
            print(f"\n✓ Position added: {position.position_id}")
            print_position_detail(position)
            
        except ValueError as e:
            print(f"\nError: Invalid input - {e}")
        except Exception as e:
            print(f"\nError: {e}")
    
    def _update_market_snapshot(self):
        """Update market snapshot (yields + TLT)."""
        print("\n--- UPDATE MARKET DATA ---\n")
        
        # Show current values if available
        current = self.market.get_latest_snapshot()
        if current:
            print(f"Current values ({current.timestamp[:16]}):")
            print(f"  TLT: ${current.tlt_price:.2f}")
            print(f"  20Y: {current.yield_20y:.3%}")
            print(f"  30Y: {current.yield_30y:.3%}")
            print()
        
        try:
            tlt = input("TLT price: $").strip()
            if not tlt:
                print("Cancelled.")
                return
            tlt_price = float(tlt)
            
            yield_20y = input("20Y yield (%): ").strip()
            yield_20y = float(yield_20y)
            
            yield_30y = input("30Y yield (%): ").strip()
            yield_30y = float(yield_30y)
            
            # Convert from percentage if needed
            if yield_20y > 1:
                yield_20y = yield_20y / 100
            if yield_30y > 1:
                yield_30y = yield_30y / 100
            
            snapshot = self.market.add_market_snapshot(
                tlt_price=tlt_price,
                yield_20y=yield_20y,
                yield_30y=yield_30y
            )
            
            print(f"\n✓ Market snapshot saved: {snapshot}")
            
        except ValueError as e:
            print(f"\nError: Invalid input - {e}")
    
    def _update_option_prices(self):
        """Update option prices."""
        print("\n--- UPDATE OPTION PRICES ---\n")
        print("Enter prices for spread legs (or 'done' to finish)\n")
        
        # Get underlying price
        snapshot = self.market.get_latest_snapshot()
        if snapshot:
            print(f"Current TLT: ${snapshot.tlt_price:.2f}")
            underlying = snapshot.tlt_price
        else:
            underlying = float(input("TLT price: $"))
        
        while True:
            print("\n" + "-" * 40)
            option_type = input("Option type (put/call) or 'done': ").strip().lower()
            
            if option_type == 'done':
                break
            
            if option_type not in ['put', 'call']:
                print("Invalid option type.")
                continue
            
            try:
                expiration = input("Expiration date (YYYY-MM-DD): ").strip()
                datetime.strptime(expiration, "%Y-%m-%d")  # Validate
                
                strike = float(input("Strike: $"))
                bid = float(input("Bid: $"))
                ask = float(input("Ask: $"))
                iv = float(input("IV (%): "))
                
                delta = input("Delta (optional): ").strip()
                delta = float(delta) if delta else None
                
                self.market.add_option_price(
                    option_type=option_type,
                    expiration_date=expiration,
                    strike=strike,
                    bid=bid,
                    ask=ask,
                    implied_vol=iv,
                    delta=delta,
                    underlying_price=underlying
                )
                
                print(f"✓ Price saved: {option_type.upper()} {strike} {expiration}")
                
            except ValueError as e:
                print(f"Error: {e}")
    
    def _mark_positions(self):
        """Mark all positions to market."""
        marks = self.monitor.mark_all_positions()
        
        if not marks:
            print("\nNo positions could be marked (check market data).")
            missing = self.monitor.get_positions_missing_data()
            if missing:
                print("\nPositions missing price data:")
                for p in missing:
                    print(f"  [{p.position_id}] {p.option_type.upper()} "
                          f"{p.short_strike}/{p.long_strike}")
            return
        
        print_position_marks(marks)
    
    def _close_position(self):
        """Close a position."""
        print("\n--- CLOSE POSITION ---\n")
        
        # Show open positions
        open_positions = self.positions.get_open_positions()
        if not open_positions:
            print("No open positions.")
            return
        
        print("Open positions:")
        for p in open_positions:
            print(f"  [{p.position_id}] {p.option_type.upper()} "
                  f"{p.short_strike}/{p.long_strike} exp {p.expiration_date}")
        
        print()
        position_id = input("Position ID to close (or 'cancel'): ").strip()
        
        if position_id.lower() == 'cancel':
            return
        
        position = self.positions.get_position(position_id)
        if not position or position.status != 'open':
            print("Position not found or already closed.")
            return
        
        # Try to get current mark
        mark = self.monitor.mark_position(position)
        
        if mark:
            print(f"\nCurrent mark:")
            print(f"  Spread value (mid): ${mark.spread_value_mid:.2f}")
            print(f"  Unrealized P&L: ${mark.unrealized_pnl_mid:+.2f}")
            
            use_mark = input("\nUse marked price for exit? (y/n) [y]: ").strip().lower()
            
            if use_mark != 'n':
                exit_reason = input("Exit reason [manual]: ").strip() or "manual"
                self.monitor.close_position_from_mark(mark, exit_reason=exit_reason)
                print(f"\n✓ Position {position_id} closed at ${mark.spread_value_mid:.2f}")
                return
        
        # Manual entry
        try:
            exit_debit = float(input("Exit debit (cost to close) per spread: $"))
            
            snapshot = self.market.get_latest_snapshot()
            if snapshot:
                exit_underlying = snapshot.tlt_price
            else:
                exit_underlying = float(input("TLT price at exit: $"))
            
            exit_reason = input("Exit reason [manual]: ").strip() or "manual"
            
            self.positions.close_position(
                position_id=position_id,
                exit_debit=exit_debit,
                exit_reason=exit_reason,
                exit_underlying=exit_underlying
            )
            
            # Refresh and show
            position = self.positions.get_position(position_id)
            print(f"\n✓ Position closed")
            print(f"  P&L: ${position.realized_pnl:+.2f}")
            
        except ValueError as e:
            print(f"\nError: {e}")
    
    def _view_positions(self):
        """View all positions."""
        print("\n--- POSITIONS ---")
        
        print("\n1. Open positions")
        print("2. Closed positions")
        print("3. All positions")
        print("4. Position detail")
        
        choice = input("\nChoice [1]: ").strip() or "1"
        
        if choice == '1':
            positions = self.positions.get_open_positions()
            print_position_summary(positions, "Open Positions")
        elif choice == '2':
            positions = self.positions.get_closed_positions()
            print_position_summary(positions, "Closed Positions")
        elif choice == '3':
            positions = self.positions.get_all_positions()
            print_position_summary(positions, "All Positions")
        elif choice == '4':
            position_id = input("Position ID: ").strip()
            position = self.positions.get_position(position_id)
            if position:
                print_position_detail(position)
            else:
                print("Position not found.")
    
    def _view_market_data(self):
        """View market data."""
        print_market_summary(self.market)
        
        # Option to view option prices
        view_options = input("\nView option prices? (y/n) [n]: ").strip().lower()
        if view_options == 'y':
            prices = self.market.get_all_option_prices()
            print_option_prices(prices)
    
    def _import_bloomberg(self):
        """Import Bloomberg CSV."""
        print("\n--- IMPORT BLOOMBERG CSV ---\n")
        
        filepath = input("Path to Bloomberg CSV: ").strip()
        
        if not filepath:
            print("Cancelled.")
            return
        
        if not Path(filepath).exists():
            print(f"File not found: {filepath}")
            return
        
        try:
            self.market.import_from_bloomberg_csv(filepath)
            print("\n✓ Import complete")
            print_market_summary(self.market)
        except Exception as e:
            print(f"\nError importing: {e}")
    
    def _configure_exit_rules(self):
        """Configure exit rules."""
        print("\n--- EXIT RULES ---\n")
        
        print(f"Current settings:")
        print(f"  Profit target: {self.exit_rules.profit_target_pct:.0%} "
              f"({'enabled' if self.exit_rules.use_profit_target else 'disabled'})")
        print(f"  Loss multiple: {self.exit_rules.loss_multiple:.1f}x "
              f"({'enabled' if self.exit_rules.use_loss_limit else 'disabled'})")
        print(f"  DTE threshold: {self.exit_rules.dte_threshold} days "
              f"({'enabled' if self.exit_rules.use_dte_threshold else 'disabled'})")
        
        print("\nEnter new values (leave blank to keep current):\n")
        
        try:
            profit = input(f"Profit target % [{self.exit_rules.profit_target_pct*100:.0f}]: ").strip()
            if profit:
                self.exit_rules.profit_target_pct = float(profit) / 100
            
            loss = input(f"Loss multiple [{self.exit_rules.loss_multiple:.1f}]: ").strip()
            if loss:
                self.exit_rules.loss_multiple = float(loss)
            
            dte = input(f"DTE threshold [{self.exit_rules.dte_threshold}]: ").strip()
            if dte:
                self.exit_rules.dte_threshold = int(dte)
            
            # Update monitor
            self.monitor.exit_rules = self.exit_rules
            
            print("\n✓ Exit rules updated")
            
        except ValueError as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(description='TLT Options Portfolio Manager')
    parser.add_argument(
        '--data-dir',
        default='.',
        help='Directory for data files (default: current directory)'
    )
    
    args = parser.parse_args()
    
    manager = PortfolioManager(data_dir=args.data_dir)
    manager.run()


if __name__ == '__main__':
    main()
