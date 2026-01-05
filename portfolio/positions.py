"""
Position Tracking

This module handles storage and management of option spread positions.
Positions are persisted to JSON for simplicity and portability.
"""

import json
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid


class PositionStatus(Enum):
    """Status of a position."""
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"


@dataclass
class SpreadPosition:
    """
    A single credit spread position.
    """
    # Identification
    position_id: str
    
    # Spread definition
    option_type: str              # 'put' or 'call'
    short_strike: float
    long_strike: float
    expiration_date: str          # ISO format YYYY-MM-DD
    
    # Entry details
    entry_date: str               # ISO format YYYY-MM-DD
    entry_credit: float           # Credit received per spread
    contracts: int                # Number of spreads
    entry_underlying: float       # TLT price at entry
    entry_short_iv: Optional[float] = None
    entry_delta: Optional[float] = None
    
    # Current state
    status: str = "open"          # open, closed, expired
    
    # Exit details (populated when closed)
    exit_date: Optional[str] = None
    exit_debit: Optional[float] = None    # Cost to close per spread
    exit_reason: Optional[str] = None     # profit_target, loss_limit, dte_threshold, manual, expiration
    exit_underlying: Optional[float] = None
    
    # Computed fields
    notes: str = ""
    
    @property
    def spread_width(self) -> float:
        return abs(self.short_strike - self.long_strike)
    
    @property
    def max_profit(self) -> float:
        """Maximum profit per spread."""
        return self.entry_credit
    
    @property
    def max_loss(self) -> float:
        """Maximum loss per spread."""
        return self.spread_width - self.entry_credit
    
    @property
    def total_credit(self) -> float:
        """Total credit received."""
        return self.entry_credit * self.contracts * 100  # Options are 100 shares
    
    @property
    def total_max_loss(self) -> float:
        """Total maximum loss."""
        return self.max_loss * self.contracts * 100
    
    @property
    def realized_pnl(self) -> Optional[float]:
        """Realized P&L if closed."""
        if self.status == "open" or self.exit_debit is None:
            return None
        pnl_per_spread = self.entry_credit - self.exit_debit
        return pnl_per_spread * self.contracts * 100
    
    @property
    def dte(self) -> int:
        """Days to expiration from today."""
        exp = datetime.strptime(self.expiration_date, "%Y-%m-%d").date()
        return (exp - date.today()).days
    
    @property
    def days_held(self) -> int:
        """Days position has been held."""
        entry = datetime.strptime(self.entry_date, "%Y-%m-%d").date()
        if self.exit_date:
            exit_dt = datetime.strptime(self.exit_date, "%Y-%m-%d").date()
            return (exit_dt - entry).days
        return (date.today() - entry).days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'position_id': self.position_id,
            'option_type': self.option_type,
            'short_strike': self.short_strike,
            'long_strike': self.long_strike,
            'expiration_date': self.expiration_date,
            'entry_date': self.entry_date,
            'entry_credit': self.entry_credit,
            'contracts': self.contracts,
            'entry_underlying': self.entry_underlying,
            'entry_short_iv': self.entry_short_iv,
            'entry_delta': self.entry_delta,
            'status': self.status,
            'exit_date': self.exit_date,
            'exit_debit': self.exit_debit,
            'exit_reason': self.exit_reason,
            'exit_underlying': self.exit_underlying,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpreadPosition':
        """Create from dictionary."""
        return cls(**data)
    
    def __repr__(self):
        type_str = "Put" if self.option_type == 'put' else "Call"
        status_str = self.status.upper()
        return (f"SpreadPosition({type_str} {self.short_strike}/{self.long_strike} "
                f"exp={self.expiration_date}, {self.contracts}x, {status_str})")


class PositionStore:
    """
    Persistent storage for positions.
    
    Stores positions in a JSON file with the structure:
    {
        "positions": [...],
        "last_updated": "ISO timestamp"
    }
    """
    
    def __init__(self, filepath: str = "positions.json"):
        self.filepath = Path(filepath)
        self._positions: Dict[str, SpreadPosition] = {}
        self._load()
    
    def _load(self):
        """Load positions from file."""
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            
            for pos_data in data.get('positions', []):
                pos = SpreadPosition.from_dict(pos_data)
                self._positions[pos.position_id] = pos
    
    def _save(self):
        """Save positions to file."""
        data = {
            'positions': [pos.to_dict() for pos in self._positions.values()],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_position(self, position: SpreadPosition) -> str:
        """Add a new position."""
        self._positions[position.position_id] = position
        self._save()
        return position.position_id
    
    def get_position(self, position_id: str) -> Optional[SpreadPosition]:
        """Get a position by ID."""
        return self._positions.get(position_id)
    
    def update_position(self, position: SpreadPosition):
        """Update an existing position."""
        if position.position_id in self._positions:
            self._positions[position.position_id] = position
            self._save()
    
    def close_position(
        self,
        position_id: str,
        exit_debit: float,
        exit_reason: str,
        exit_underlying: float,
        exit_date: Optional[str] = None
    ):
        """Close a position."""
        pos = self._positions.get(position_id)
        if pos is None:
            raise ValueError(f"Position {position_id} not found")
        
        pos.status = "closed"
        pos.exit_date = exit_date or date.today().isoformat()
        pos.exit_debit = exit_debit
        pos.exit_reason = exit_reason
        pos.exit_underlying = exit_underlying
        
        self._save()
    
    def get_open_positions(self) -> List[SpreadPosition]:
        """Get all open positions."""
        return [p for p in self._positions.values() if p.status == "open"]
    
    def get_closed_positions(self) -> List[SpreadPosition]:
        """Get all closed positions."""
        return [p for p in self._positions.values() if p.status != "open"]
    
    def get_all_positions(self) -> List[SpreadPosition]:
        """Get all positions."""
        return list(self._positions.values())
    
    def get_positions_by_expiration(self, expiration_date: str) -> List[SpreadPosition]:
        """Get positions expiring on a specific date."""
        return [p for p in self._positions.values() 
                if p.expiration_date == expiration_date]
    
    def delete_position(self, position_id: str):
        """Delete a position (use with caution)."""
        if position_id in self._positions:
            del self._positions[position_id]
            self._save()


def create_position(
    option_type: str,
    short_strike: float,
    long_strike: float,
    expiration_date: str,
    entry_credit: float,
    contracts: int,
    entry_underlying: float,
    entry_short_iv: Optional[float] = None,
    entry_delta: Optional[float] = None,
    entry_date: Optional[str] = None,
    notes: str = ""
) -> SpreadPosition:
    """
    Create a new position with auto-generated ID.
    
    Parameters
    ----------
    option_type : str
        'put' or 'call'
    short_strike : float
        Strike of the short option
    long_strike : float
        Strike of the long option
    expiration_date : str
        Expiration date (YYYY-MM-DD)
    entry_credit : float
        Credit received per spread
    contracts : int
        Number of spreads
    entry_underlying : float
        TLT price at entry
    entry_short_iv : float, optional
        IV of short strike at entry
    entry_delta : float, optional
        Delta of short strike at entry
    entry_date : str, optional
        Entry date (defaults to today)
    notes : str
        Optional notes
        
    Returns
    -------
    SpreadPosition
        New position object
    """
    position_id = str(uuid.uuid4())[:8]  # Short UUID
    
    return SpreadPosition(
        position_id=position_id,
        option_type=option_type.lower(),
        short_strike=short_strike,
        long_strike=long_strike,
        expiration_date=expiration_date,
        entry_date=entry_date or date.today().isoformat(),
        entry_credit=entry_credit,
        contracts=contracts,
        entry_underlying=entry_underlying,
        entry_short_iv=entry_short_iv,
        entry_delta=entry_delta,
        notes=notes
    )


def print_position_summary(positions: List[SpreadPosition], title: str = "Positions"):
    """Print a summary table of positions."""
    print("\n" + "=" * 100)
    print(title.upper())
    print("=" * 100)
    
    if not positions:
        print("\nNo positions found.")
        return
    
    print(f"\n{'ID':<10} {'Type':<6} {'Strikes':<12} {'Exp':<12} {'DTE':<6} "
          f"{'Contracts':<10} {'Credit':<10} {'Status':<10}")
    print("-" * 100)
    
    for p in sorted(positions, key=lambda x: x.expiration_date):
        type_str = "Put" if p.option_type == 'put' else "Call"
        strikes = f"{p.short_strike}/{p.long_strike}"
        dte = p.dte if p.status == "open" else "-"
        
        print(f"{p.position_id:<10} {type_str:<6} {strikes:<12} {p.expiration_date:<12} "
              f"{str(dte):<6} {p.contracts:<10} ${p.entry_credit:<9.2f} {p.status:<10}")
    
    # Summary stats
    open_positions = [p for p in positions if p.status == "open"]
    if open_positions:
        total_credit = sum(p.total_credit for p in open_positions)
        total_max_loss = sum(p.total_max_loss for p in open_positions)
        total_contracts = sum(p.contracts for p in open_positions)
        
        print("-" * 100)
        print(f"{'OPEN TOTALS':<10} {'':<6} {'':<12} {'':<12} {'':<6} "
              f"{total_contracts:<10} ${total_credit:<9.2f}")
        print(f"Max Loss at Risk: ${total_max_loss:.2f}")
    
    print("=" * 100)


def print_position_detail(position: SpreadPosition):
    """Print detailed view of a single position."""
    print("\n" + "=" * 60)
    print(f"POSITION: {position.position_id}")
    print("=" * 60)
    
    type_str = "Bull Put Spread" if position.option_type == 'put' else "Bear Call Spread"
    
    print(f"\nType: {type_str}")
    print(f"Strikes: {position.short_strike}/{position.long_strike} (width: ${position.spread_width:.2f})")
    print(f"Expiration: {position.expiration_date} ({position.dte} DTE)")
    print(f"Contracts: {position.contracts}")
    
    print(f"\nEntry Details:")
    print(f"  Date: {position.entry_date}")
    print(f"  Credit: ${position.entry_credit:.2f} per spread")
    print(f"  Total Credit: ${position.total_credit:.2f}")
    print(f"  Underlying: ${position.entry_underlying:.2f}")
    if position.entry_short_iv:
        print(f"  Short IV: {position.entry_short_iv:.1%}")
    if position.entry_delta:
        print(f"  Short Delta: {position.entry_delta:.3f}")
    
    print(f"\nRisk:")
    print(f"  Max Profit: ${position.max_profit:.2f} per spread (${position.max_profit * position.contracts * 100:.2f} total)")
    print(f"  Max Loss: ${position.max_loss:.2f} per spread (${position.total_max_loss:.2f} total)")
    
    print(f"\nStatus: {position.status.upper()}")
    print(f"Days Held: {position.days_held}")
    
    if position.status != "open":
        print(f"\nExit Details:")
        print(f"  Date: {position.exit_date}")
        print(f"  Debit: ${position.exit_debit:.2f} per spread")
        print(f"  Reason: {position.exit_reason}")
        print(f"  Underlying: ${position.exit_underlying:.2f}")
        print(f"  Realized P&L: ${position.realized_pnl:.2f}")
    
    if position.notes:
        print(f"\nNotes: {position.notes}")
    
    print("=" * 60)
