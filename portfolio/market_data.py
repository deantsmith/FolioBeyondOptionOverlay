"""
Market Data Storage

This module handles storage and retrieval of market data updates.
Supports ad hoc updates for:
- Treasury yields (20Y, 30Y)
- TLT price
- Option prices for specific contracts

Data is persisted to JSON files for simplicity.
"""

import json
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


@dataclass
class MarketSnapshot:
    """
    A point-in-time snapshot of market data.
    """
    timestamp: str                    # ISO format datetime
    
    # Underlying data
    tlt_price: float
    yield_20y: float                  # Decimal (0.0450 = 4.50%)
    yield_30y: float
    
    # Optional metadata
    source: str = "manual"            # manual, bloomberg, etc.
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'tlt_price': self.tlt_price,
            'yield_20y': self.yield_20y,
            'yield_30y': self.yield_30y,
            'source': self.source,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketSnapshot':
        return cls(**data)
    
    def __repr__(self):
        return (f"MarketSnapshot({self.timestamp[:10]}, TLT=${self.tlt_price:.2f}, "
                f"20Y={self.yield_20y:.3%}, 30Y={self.yield_30y:.3%})")


@dataclass
class OptionPriceUpdate:
    """
    Price update for a specific option contract.
    """
    timestamp: str
    
    # Contract identifier
    option_type: str                  # 'put' or 'call'
    expiration_date: str              # YYYY-MM-DD
    strike: float
    
    # Prices
    bid: float
    ask: float
    implied_vol: float                # Decimal (0.162 = 16.2%)
    
    # Greeks (optional)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    
    # Underlying at time of quote
    underlying_price: Optional[float] = None
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def contract_key(self) -> str:
        """Unique key for this contract."""
        return f"{self.option_type}_{self.expiration_date}_{self.strike}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'option_type': self.option_type,
            'expiration_date': self.expiration_date,
            'strike': self.strike,
            'bid': self.bid,
            'ask': self.ask,
            'implied_vol': self.implied_vol,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'underlying_price': self.underlying_price
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptionPriceUpdate':
        return cls(**data)
    
    def __repr__(self):
        type_str = "P" if self.option_type == 'put' else "C"
        return (f"OptionPrice({type_str}{self.strike} {self.expiration_date[:10]}, "
                f"bid={self.bid:.2f}, ask={self.ask:.2f}, IV={self.implied_vol:.1%})")


class MarketDataStore:
    """
    Persistent storage for market data.
    
    Maintains two data sets:
    1. Market snapshots (yields, TLT price) - time series
    2. Option prices - latest quote per contract
    
    File structure:
    {
        "market_snapshots": [...],
        "option_prices": {...},  # keyed by contract
        "last_updated": "ISO timestamp"
    }
    """
    
    def __init__(self, filepath: str = "market_data.json"):
        self.filepath = Path(filepath)
        self._snapshots: List[MarketSnapshot] = []
        self._option_prices: Dict[str, OptionPriceUpdate] = {}  # Latest price per contract
        self._option_history: Dict[str, List[OptionPriceUpdate]] = {}  # Full history
        self._load()
    
    def _load(self):
        """Load data from file."""
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            
            # Load snapshots
            for snap_data in data.get('market_snapshots', []):
                self._snapshots.append(MarketSnapshot.from_dict(snap_data))
            
            # Load option prices (latest)
            for key, opt_data in data.get('option_prices', {}).items():
                self._option_prices[key] = OptionPriceUpdate.from_dict(opt_data)
            
            # Load option history
            for key, history in data.get('option_history', {}).items():
                self._option_history[key] = [
                    OptionPriceUpdate.from_dict(h) for h in history
                ]
    
    def _save(self):
        """Save data to file."""
        data = {
            'market_snapshots': [s.to_dict() for s in self._snapshots],
            'option_prices': {k: v.to_dict() for k, v in self._option_prices.items()},
            'option_history': {
                k: [h.to_dict() for h in v] 
                for k, v in self._option_history.items()
            },
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    # =========================================================================
    # Market Snapshots (Yields + TLT)
    # =========================================================================
    
    def add_market_snapshot(
        self,
        tlt_price: float,
        yield_20y: float,
        yield_30y: float,
        timestamp: Optional[str] = None,
        source: str = "manual",
        notes: str = ""
    ) -> MarketSnapshot:
        """
        Add a market data snapshot.
        
        Parameters
        ----------
        tlt_price : float
            TLT price
        yield_20y : float
            20-year yield (decimal, e.g., 0.0450)
        yield_30y : float
            30-year yield (decimal)
        timestamp : str, optional
            ISO timestamp (defaults to now)
        source : str
            Data source
        notes : str
            Optional notes
            
        Returns
        -------
        MarketSnapshot
            The created snapshot
        """
        # Convert percentage yields to decimal if needed
        if yield_20y > 0.5:  # Likely percentage
            yield_20y = yield_20y / 100.0
        if yield_30y > 0.5:
            yield_30y = yield_30y / 100.0
        
        snapshot = MarketSnapshot(
            timestamp=timestamp or datetime.now().isoformat(),
            tlt_price=tlt_price,
            yield_20y=yield_20y,
            yield_30y=yield_30y,
            source=source,
            notes=notes
        )
        
        self._snapshots.append(snapshot)
        self._snapshots.sort(key=lambda x: x.timestamp)
        self._save()
        
        return snapshot
    
    def get_latest_snapshot(self) -> Optional[MarketSnapshot]:
        """Get the most recent market snapshot."""
        if not self._snapshots:
            return None
        return self._snapshots[-1]
    
    def get_snapshots(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[MarketSnapshot]:
        """Get snapshots within a date range."""
        snapshots = self._snapshots
        
        if start_date:
            snapshots = [s for s in snapshots if s.timestamp[:10] >= start_date]
        if end_date:
            snapshots = [s for s in snapshots if s.timestamp[:10] <= end_date]
        
        return snapshots
    
    # =========================================================================
    # Option Prices
    # =========================================================================
    
    def add_option_price(
        self,
        option_type: str,
        expiration_date: str,
        strike: float,
        bid: float,
        ask: float,
        implied_vol: float,
        delta: Optional[float] = None,
        gamma: Optional[float] = None,
        theta: Optional[float] = None,
        vega: Optional[float] = None,
        underlying_price: Optional[float] = None,
        timestamp: Optional[str] = None
    ) -> OptionPriceUpdate:
        """
        Add or update an option price.
        
        Parameters
        ----------
        option_type : str
            'put' or 'call'
        expiration_date : str
            Expiration date (YYYY-MM-DD)
        strike : float
            Strike price
        bid : float
            Bid price
        ask : float
            Ask price
        implied_vol : float
            Implied volatility (percentage, e.g., 16.2, will be converted to 0.162)
        delta : float, optional
            Delta
        gamma : float, optional
            Gamma
        theta : float, optional
            Theta
        vega : float, optional
            Vega
        underlying_price : float, optional
            TLT price at time of quote
        timestamp : str, optional
            ISO timestamp (defaults to now)
            
        Returns
        -------
        OptionPriceUpdate
            The created/updated price
        """
        # Convert IV from percentage to decimal if needed
        if implied_vol > 1.0:
            implied_vol = implied_vol / 100.0
        
        price_update = OptionPriceUpdate(
            timestamp=timestamp or datetime.now().isoformat(),
            option_type=option_type.lower(),
            expiration_date=expiration_date,
            strike=strike,
            bid=bid,
            ask=ask,
            implied_vol=implied_vol,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            underlying_price=underlying_price
        )
        
        key = price_update.contract_key
        
        # Store as latest
        self._option_prices[key] = price_update
        
        # Add to history
        if key not in self._option_history:
            self._option_history[key] = []
        self._option_history[key].append(price_update)
        
        self._save()
        
        return price_update
    
    def get_option_price(
        self,
        option_type: str,
        expiration_date: str,
        strike: float
    ) -> Optional[OptionPriceUpdate]:
        """Get the latest price for a specific contract."""
        key = f"{option_type.lower()}_{expiration_date}_{strike}"
        return self._option_prices.get(key)
    
    def get_spread_prices(
        self,
        option_type: str,
        expiration_date: str,
        short_strike: float,
        long_strike: float
    ) -> Tuple[Optional[OptionPriceUpdate], Optional[OptionPriceUpdate]]:
        """Get prices for both legs of a spread."""
        short = self.get_option_price(option_type, expiration_date, short_strike)
        long = self.get_option_price(option_type, expiration_date, long_strike)
        return short, long
    
    def get_all_option_prices(self) -> List[OptionPriceUpdate]:
        """Get all latest option prices."""
        return list(self._option_prices.values())
    
    def get_option_prices_by_expiration(
        self,
        expiration_date: str
    ) -> List[OptionPriceUpdate]:
        """Get all option prices for a specific expiration."""
        return [
            p for p in self._option_prices.values()
            if p.expiration_date == expiration_date
        ]
    
    def get_option_history(
        self,
        option_type: str,
        expiration_date: str,
        strike: float
    ) -> List[OptionPriceUpdate]:
        """Get price history for a specific contract."""
        key = f"{option_type.lower()}_{expiration_date}_{strike}"
        return self._option_history.get(key, [])
    
    # =========================================================================
    # Bulk Updates
    # =========================================================================
    
    def bulk_add_option_prices(
        self,
        prices: List[Dict[str, Any]],
        underlying_price: Optional[float] = None,
        timestamp: Optional[str] = None
    ):
        """
        Add multiple option prices at once.
        
        Parameters
        ----------
        prices : List[Dict]
            List of price dictionaries with keys:
            option_type, expiration_date, strike, bid, ask, implied_vol,
            and optionally delta, gamma, theta, vega
        underlying_price : float, optional
            TLT price (applied to all if not in individual dicts)
        timestamp : str, optional
            Timestamp (applied to all if not in individual dicts)
        """
        ts = timestamp or datetime.now().isoformat()
        
        for p in prices:
            self.add_option_price(
                option_type=p['option_type'],
                expiration_date=p['expiration_date'],
                strike=p['strike'],
                bid=p['bid'],
                ask=p['ask'],
                implied_vol=p['implied_vol'],
                delta=p.get('delta'),
                gamma=p.get('gamma'),
                theta=p.get('theta'),
                vega=p.get('vega'),
                underlying_price=p.get('underlying_price', underlying_price),
                timestamp=p.get('timestamp', ts)
            )
    
    def import_from_bloomberg_csv(self, filepath: str):
        """
        Import option prices from Bloomberg CSV format.
        
        Uses the same format as the bloomberg.py loader.
        """
        import pandas as pd
        
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Get underlying and timestamp from first row
        underlying = df['underlying_price'].iloc[0]
        timestamp = datetime.now().isoformat()
        
        if 'quote_date' in df.columns:
            quote_date = df['quote_date'].iloc[0]
            if 'quote_time' in df.columns:
                quote_time = df['quote_time'].iloc[0]
                timestamp = f"{quote_date}T{quote_time}"
            else:
                timestamp = f"{quote_date}T16:00:00"
        
        # Also add market snapshot
        # We need yields for this - check if they're in the data or get from latest
        latest = self.get_latest_snapshot()
        if latest:
            self.add_market_snapshot(
                tlt_price=underlying,
                yield_20y=latest.yield_20y,
                yield_30y=latest.yield_30y,
                timestamp=timestamp,
                source="bloomberg_import"
            )
        
        # Add option prices
        for _, row in df.iterrows():
            iv = row['implied_vol']
            if iv > 1.0:  # Percentage form
                iv = iv / 100.0
            
            self.add_option_price(
                option_type=row['option_type'].lower(),
                expiration_date=str(row['expiration_date'])[:10],
                strike=row['strike'],
                bid=row['bid'],
                ask=row['ask'],
                implied_vol=iv,
                delta=row.get('delta'),
                gamma=row.get('gamma'),
                theta=row.get('theta'),
                vega=row.get('vega'),
                underlying_price=underlying,
                timestamp=timestamp
            )
        
        print(f"Imported {len(df)} option prices from {filepath}")


def print_market_summary(store: MarketDataStore):
    """Print summary of stored market data."""
    print("\n" + "=" * 70)
    print("MARKET DATA SUMMARY")
    print("=" * 70)
    
    # Latest snapshot
    latest = store.get_latest_snapshot()
    if latest:
        print(f"\nLatest Market Snapshot ({latest.timestamp[:16]}):")
        print(f"  TLT: ${latest.tlt_price:.2f}")
        print(f"  20Y Yield: {latest.yield_20y:.3%}")
        print(f"  30Y Yield: {latest.yield_30y:.3%}")
    else:
        print("\nNo market snapshots available.")
    
    # Option prices
    options = store.get_all_option_prices()
    if options:
        # Group by expiration
        expirations = {}
        for opt in options:
            if opt.expiration_date not in expirations:
                expirations[opt.expiration_date] = []
            expirations[opt.expiration_date].append(opt)
        
        print(f"\nOption Prices ({len(options)} contracts across {len(expirations)} expirations):")
        
        for exp in sorted(expirations.keys()):
            opts = expirations[exp]
            puts = [o for o in opts if o.option_type == 'put']
            calls = [o for o in opts if o.option_type == 'call']
            print(f"  {exp}: {len(puts)} puts, {len(calls)} calls")
    else:
        print("\nNo option prices stored.")
    
    # Snapshot history
    snapshots = store.get_snapshots()
    if len(snapshots) > 1:
        print(f"\nMarket History: {len(snapshots)} snapshots")
        print(f"  From: {snapshots[0].timestamp[:10]}")
        print(f"  To:   {snapshots[-1].timestamp[:10]}")
    
    print("=" * 70)


def print_option_prices(prices: List[OptionPriceUpdate], title: str = "Option Prices"):
    """Print option prices table."""
    if not prices:
        print(f"\n{title}: No prices available")
        return
    
    print("\n" + "=" * 90)
    print(title.upper())
    print("=" * 90)
    
    print(f"\n{'Type':<6} {'Strike':<10} {'Exp':<12} {'Bid':<10} {'Ask':<10} "
          f"{'Mid':<10} {'IV':<10} {'Delta':<10}")
    print("-" * 90)
    
    for p in sorted(prices, key=lambda x: (x.expiration_date, x.option_type, x.strike)):
        type_str = "Put" if p.option_type == 'put' else "Call"
        delta_str = f"{p.delta:.3f}" if p.delta else "-"
        
        print(f"{type_str:<6} ${p.strike:<9.2f} {p.expiration_date:<12} "
              f"${p.bid:<9.2f} ${p.ask:<9.2f} ${p.mid:<9.2f} "
              f"{p.implied_vol:.1%}     {delta_str:<10}")
    
    print("=" * 90)
