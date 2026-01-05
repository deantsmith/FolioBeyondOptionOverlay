"""
Bloomberg Option Data Integration

This module handles:
1. Loading option chain data from Bloomberg CSV exports
2. Building implied volatility surfaces
3. Pricing spreads with actual market quotes
4. Filtering for liquid contracts
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime, date
from pathlib import Path


@dataclass
class OptionContract:
    """Single option contract from Bloomberg."""
    
    quote_date: date
    quote_time: Optional[str]
    underlying_price: float
    option_type: str  # 'put' or 'call'
    expiration_date: date
    strike: float
    bid: float
    ask: float
    implied_vol: float  # Decimal (0.162, not 16.2)
    delta: float
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    
    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as percentage of mid."""
        if self.mid > 0:
            return self.spread / self.mid
        return float('inf')
    
    @property
    def dte(self) -> int:
        """Days to expiration."""
        return (self.expiration_date - self.quote_date).days
    
    @property
    def moneyness(self) -> float:
        """Strike / Underlying price."""
        return self.strike / self.underlying_price
    
    def __repr__(self):
        return (f"OptionContract({self.option_type.upper()} {self.strike} "
                f"{self.expiration_date} bid={self.bid:.2f} ask={self.ask:.2f} "
                f"IV={self.implied_vol:.1%})")


@dataclass
class OptionChain:
    """Collection of option contracts for a single underlying."""
    
    quote_date: date
    quote_time: Optional[str]
    underlying_price: float
    contracts: List[OptionContract]
    
    # Computed on demand
    _puts: Optional[List[OptionContract]] = field(default=None, repr=False)
    _calls: Optional[List[OptionContract]] = field(default=None, repr=False)
    _expirations: Optional[List[date]] = field(default=None, repr=False)
    _strikes: Optional[List[float]] = field(default=None, repr=False)
    
    @property
    def puts(self) -> List[OptionContract]:
        """All put contracts."""
        if self._puts is None:
            self._puts = [c for c in self.contracts if c.option_type == 'put']
        return self._puts
    
    @property
    def calls(self) -> List[OptionContract]:
        """All call contracts."""
        if self._calls is None:
            self._calls = [c for c in self.contracts if c.option_type == 'call']
        return self._calls
    
    @property
    def expirations(self) -> List[date]:
        """Unique expiration dates, sorted."""
        if self._expirations is None:
            self._expirations = sorted(set(c.expiration_date for c in self.contracts))
        return self._expirations
    
    @property
    def strikes(self) -> List[float]:
        """Unique strikes, sorted."""
        if self._strikes is None:
            self._strikes = sorted(set(c.strike for c in self.contracts))
        return self._strikes
    
    def get_contract(
        self,
        option_type: str,
        expiration: date,
        strike: float
    ) -> Optional[OptionContract]:
        """Find a specific contract."""
        for c in self.contracts:
            if (c.option_type == option_type and 
                c.expiration_date == expiration and 
                c.strike == strike):
                return c
        return None
    
    def get_contracts_by_expiration(self, expiration: date) -> List[OptionContract]:
        """Get all contracts for a specific expiration."""
        return [c for c in self.contracts if c.expiration_date == expiration]
    
    def get_contracts_by_dte_range(
        self,
        min_dte: int,
        max_dte: int
    ) -> List[OptionContract]:
        """Get contracts within a DTE range."""
        return [c for c in self.contracts if min_dte <= c.dte <= max_dte]
    
    def filter_liquid(
        self,
        min_volume: int = 0,
        min_open_interest: int = 100,
        max_spread_pct: float = 0.20
    ) -> 'OptionChain':
        """Return a new chain with only liquid contracts."""
        filtered = []
        for c in self.contracts:
            # Check volume
            if c.volume is not None and c.volume < min_volume:
                continue
            # Check OI
            if c.open_interest is not None and c.open_interest < min_open_interest:
                continue
            # Check spread
            if c.spread_pct > max_spread_pct:
                continue
            filtered.append(c)
        
        return OptionChain(
            quote_date=self.quote_date,
            quote_time=self.quote_time,
            underlying_price=self.underlying_price,
            contracts=filtered
        )
    
    def __repr__(self):
        return (f"OptionChain({self.quote_date}, TLT=${self.underlying_price:.2f}, "
                f"{len(self.contracts)} contracts, {len(self.expirations)} expirations)")


def load_bloomberg_chain(
    filepath: Union[str, Path],
    iv_as_percentage: bool = True
) -> OptionChain:
    """
    Load option chain from Bloomberg CSV export.
    
    Parameters
    ----------
    filepath : str or Path
        Path to CSV file.
    iv_as_percentage : bool
        If True, IV is in percentage form (16.2) and will be converted to decimal.
        
    Returns
    -------
    OptionChain
        Loaded option chain.
    """
    df = pd.read_csv(filepath)
    
    # Standardize column names (lowercase, strip whitespace)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Parse dates
    df['quote_date'] = pd.to_datetime(df['quote_date']).dt.date
    df['expiration_date'] = pd.to_datetime(df['expiration_date']).dt.date
    
    # Standardize option type
    df['option_type'] = df['option_type'].str.lower().str.strip()
    
    # Convert IV from percentage to decimal if needed
    if iv_as_percentage:
        df['implied_vol'] = df['implied_vol'] / 100.0
    
    # Get quote metadata from first row
    quote_date = df['quote_date'].iloc[0]
    quote_time = df['quote_time'].iloc[0] if 'quote_time' in df.columns else None
    underlying_price = df['underlying_price'].iloc[0]
    
    # Build contracts
    contracts = []
    for _, row in df.iterrows():
        contract = OptionContract(
            quote_date=row['quote_date'],
            quote_time=row.get('quote_time'),
            underlying_price=row['underlying_price'],
            option_type=row['option_type'],
            expiration_date=row['expiration_date'],
            strike=row['strike'],
            bid=row['bid'],
            ask=row['ask'],
            implied_vol=row['implied_vol'],
            delta=row['delta'],
            gamma=row.get('gamma'),
            theta=row.get('theta'),
            vega=row.get('vega'),
            volume=row.get('volume'),
            open_interest=row.get('open_interest')
        )
        contracts.append(contract)
    
    return OptionChain(
        quote_date=quote_date,
        quote_time=quote_time,
        underlying_price=underlying_price,
        contracts=contracts
    )


@dataclass
class IVSurface:
    """
    Implied volatility surface for interpolation.
    
    Provides IV estimates for any strike/expiration within the surface bounds.
    """
    
    quote_date: date
    underlying_price: float
    
    # Raw data points
    put_ivs: Dict[Tuple[date, float], float] = field(default_factory=dict)
    call_ivs: Dict[Tuple[date, float], float] = field(default_factory=dict)
    
    # Interpolators (built on demand)
    _put_interp: Optional[callable] = field(default=None, repr=False)
    _call_interp: Optional[callable] = field(default=None, repr=False)
    
    @classmethod
    def from_chain(cls, chain: OptionChain) -> 'IVSurface':
        """Build IV surface from option chain."""
        surface = cls(
            quote_date=chain.quote_date,
            underlying_price=chain.underlying_price
        )
        
        for c in chain.contracts:
            key = (c.expiration_date, c.strike)
            if c.option_type == 'put':
                surface.put_ivs[key] = c.implied_vol
            else:
                surface.call_ivs[key] = c.implied_vol
        
        return surface
    
    def _build_interpolator(self, ivs: Dict[Tuple[date, float], float]) -> callable:
        """Build 2D interpolator from IV points."""
        if len(ivs) < 4:
            # Not enough points for 2D interpolation, return constant
            avg_iv = np.mean(list(ivs.values())) if ivs else 0.15
            return lambda dte, strike: avg_iv
        
        # Convert to arrays
        dtes = []
        strikes = []
        vols = []
        
        for (exp_date, strike), iv in ivs.items():
            dte = (exp_date - self.quote_date).days
            dtes.append(dte)
            strikes.append(strike)
            vols.append(iv)
        
        dtes = np.array(dtes)
        strikes = np.array(strikes)
        vols = np.array(vols)
        
        # Use linear interpolation with nearest extrapolation
        try:
            interp = interpolate.LinearNDInterpolator(
                list(zip(dtes, strikes)), vols,
                fill_value=np.nan
            )
            
            # Wrapper that handles extrapolation
            nearest = interpolate.NearestNDInterpolator(
                list(zip(dtes, strikes)), vols
            )
            
            def interpolator(dte, strike):
                result = interp(dte, strike)
                if np.isnan(result):
                    result = nearest(dte, strike)
                return float(result)
            
            return interpolator
            
        except Exception:
            # Fall back to average
            avg_iv = np.mean(vols)
            return lambda dte, strike: avg_iv
    
    def get_put_iv(self, expiration: date, strike: float) -> float:
        """Get interpolated put IV."""
        # Check for exact match first
        key = (expiration, strike)
        if key in self.put_ivs:
            return self.put_ivs[key]
        
        # Build interpolator if needed
        if self._put_interp is None:
            self._put_interp = self._build_interpolator(self.put_ivs)
        
        dte = (expiration - self.quote_date).days
        return self._put_interp(dte, strike)
    
    def get_call_iv(self, expiration: date, strike: float) -> float:
        """Get interpolated call IV."""
        key = (expiration, strike)
        if key in self.call_ivs:
            return self.call_ivs[key]
        
        if self._call_interp is None:
            self._call_interp = self._build_interpolator(self.call_ivs)
        
        dte = (expiration - self.quote_date).days
        return self._call_interp(dte, strike)
    
    def get_iv(self, option_type: str, expiration: date, strike: float) -> float:
        """Get IV for any option."""
        if option_type == 'put':
            return self.get_put_iv(expiration, strike)
        else:
            return self.get_call_iv(expiration, strike)
    
    def get_iv_by_dte(self, option_type: str, dte: int, strike: float) -> float:
        """Get IV by days to expiration."""
        expiration = self.quote_date + pd.Timedelta(days=dte)
        if isinstance(expiration, pd.Timestamp):
            expiration = expiration.date()
        return self.get_iv(option_type, expiration, strike)
    
    def get_atm_iv(self, expiration: date) -> float:
        """Get ATM implied volatility (average of put and call at nearest strike)."""
        # Find nearest strike to underlying
        all_strikes = set(k[1] for k in self.put_ivs.keys()) | set(k[1] for k in self.call_ivs.keys())
        if not all_strikes:
            return 0.15  # Default
        
        nearest_strike = min(all_strikes, key=lambda x: abs(x - self.underlying_price))
        
        put_iv = self.get_put_iv(expiration, nearest_strike)
        call_iv = self.get_call_iv(expiration, nearest_strike)
        
        return (put_iv + call_iv) / 2


@dataclass
class MarketSpreadQuote:
    """Spread quote using actual market prices."""
    
    # Spread definition
    option_type: str  # 'put' or 'call'
    short_strike: float
    long_strike: float
    expiration_date: date
    
    # Individual leg quotes
    short_contract: OptionContract
    long_contract: OptionContract
    
    # Spread pricing (realistic fills)
    credit_at_bid: float      # Sell short at bid, buy long at ask
    credit_at_mid: float      # Mid prices
    credit_at_ask: float      # Best case: sell at ask, buy at bid
    
    # Greeks (net)
    net_delta: float
    net_gamma: Optional[float]
    net_theta: Optional[float]
    net_vega: Optional[float]
    
    # Risk
    max_loss_at_bid: float    # Width - credit_at_bid
    
    @property
    def width(self) -> float:
        return abs(self.short_strike - self.long_strike)
    
    @property
    def dte(self) -> int:
        return self.short_contract.dte
    
    def __repr__(self):
        type_str = "Put" if self.option_type == 'put' else "Call"
        return (f"MarketSpreadQuote({type_str} {self.short_strike}/{self.long_strike} "
                f"{self.expiration_date}, credit=${self.credit_at_bid:.2f}-${self.credit_at_ask:.2f})")


def price_spread_from_chain(
    chain: OptionChain,
    option_type: str,
    short_strike: float,
    long_strike: float,
    expiration: date
) -> Optional[MarketSpreadQuote]:
    """
    Price a credit spread using actual market quotes.
    
    Parameters
    ----------
    chain : OptionChain
        Option chain with market data.
    option_type : str
        'put' or 'call'.
    short_strike : float
        Strike of the short option.
    long_strike : float
        Strike of the long option.
    expiration : date
        Expiration date.
        
    Returns
    -------
    MarketSpreadQuote or None
        Spread quote, or None if contracts not found.
    """
    short_contract = chain.get_contract(option_type, expiration, short_strike)
    long_contract = chain.get_contract(option_type, expiration, long_strike)
    
    if short_contract is None or long_contract is None:
        return None
    
    # Credit spread pricing:
    # Realistic: sell short at bid, buy long at ask
    # Mid: both at mid
    # Best case: sell at ask, buy at bid
    credit_at_bid = short_contract.bid - long_contract.ask
    credit_at_mid = short_contract.mid - long_contract.mid
    credit_at_ask = short_contract.ask - long_contract.bid
    
    # Net Greeks (we're short the short, long the long)
    net_delta = -short_contract.delta + long_contract.delta
    
    net_gamma = None
    if short_contract.gamma is not None and long_contract.gamma is not None:
        net_gamma = -short_contract.gamma + long_contract.gamma
    
    net_theta = None
    if short_contract.theta is not None and long_contract.theta is not None:
        net_theta = -short_contract.theta + long_contract.theta
    
    net_vega = None
    if short_contract.vega is not None and long_contract.vega is not None:
        net_vega = -short_contract.vega + long_contract.vega
    
    width = abs(short_strike - long_strike)
    max_loss = width - credit_at_bid
    
    return MarketSpreadQuote(
        option_type=option_type,
        short_strike=short_strike,
        long_strike=long_strike,
        expiration_date=expiration,
        short_contract=short_contract,
        long_contract=long_contract,
        credit_at_bid=credit_at_bid,
        credit_at_mid=credit_at_mid,
        credit_at_ask=credit_at_ask,
        net_delta=net_delta,
        net_gamma=net_gamma,
        net_theta=net_theta,
        net_vega=net_vega,
        max_loss_at_bid=max_loss
    )


def find_spread_by_delta(
    chain: OptionChain,
    option_type: str,
    expiration: date,
    target_delta: float,
    spread_width: float,
    delta_tolerance: float = 0.05
) -> Optional[MarketSpreadQuote]:
    """
    Find a spread with short strike near target delta.
    
    Parameters
    ----------
    chain : OptionChain
        Option chain.
    option_type : str
        'put' or 'call'.
    expiration : date
        Target expiration.
    target_delta : float
        Target delta for short strike (absolute value).
    spread_width : float
        Desired spread width.
    delta_tolerance : float
        Acceptable deviation from target delta.
        
    Returns
    -------
    MarketSpreadQuote or None
        Best matching spread, or None.
    """
    # Get contracts for this expiration
    contracts = [c for c in chain.contracts 
                 if c.expiration_date == expiration and c.option_type == option_type]
    
    if not contracts:
        return None
    
    # For puts, delta is negative; for calls, positive
    if option_type == 'put':
        target_delta = -abs(target_delta)
    else:
        target_delta = abs(target_delta)
    
    # Find contract closest to target delta
    best_short = min(contracts, key=lambda c: abs(c.delta - target_delta))
    
    if abs(best_short.delta - target_delta) > delta_tolerance:
        return None
    
    # Find long strike
    if option_type == 'put':
        long_strike = best_short.strike - spread_width
    else:
        long_strike = best_short.strike + spread_width
    
    # Check if long strike exists
    long_contract = chain.get_contract(option_type, expiration, long_strike)
    if long_contract is None:
        # Try nearest available strike
        available_strikes = [c.strike for c in contracts]
        if option_type == 'put':
            valid_strikes = [s for s in available_strikes if s < best_short.strike]
        else:
            valid_strikes = [s for s in available_strikes if s > best_short.strike]
        
        if not valid_strikes:
            return None
        
        long_strike = min(valid_strikes, key=lambda s: abs(abs(s - best_short.strike) - spread_width))
    
    return price_spread_from_chain(
        chain, option_type, best_short.strike, long_strike, expiration
    )


def generate_market_spread_candidates(
    chain: OptionChain,
    min_dte: int = 30,
    max_dte: int = 90,
    delta_targets: List[float] = [0.15, 0.20, 0.25, 0.30],
    spread_widths: List[float] = [2, 3, 5],
    option_types: List[str] = ['put', 'call'],
    min_credit: float = 0.20,
    min_open_interest: int = 100
) -> List[MarketSpreadQuote]:
    """
    Generate spread candidates from market data.
    
    Parameters
    ----------
    chain : OptionChain
        Option chain.
    min_dte, max_dte : int
        DTE range to consider.
    delta_targets : List[float]
        Target deltas for short strikes.
    spread_widths : List[float]
        Spread widths to consider.
    option_types : List[str]
        Option types to include.
    min_credit : float
        Minimum credit to consider (at bid).
    min_open_interest : int
        Minimum OI for liquidity.
        
    Returns
    -------
    List[MarketSpreadQuote]
        Valid spread candidates (deduplicated).
    """
    # Filter chain for liquidity
    liquid_chain = chain.filter_liquid(min_open_interest=min_open_interest)
    
    # Get valid expirations
    valid_expirations = [
        exp for exp in liquid_chain.expirations
        if min_dte <= (exp - chain.quote_date).days <= max_dte
    ]
    
    candidates = []
    seen = set()  # Track unique spreads: (type, exp, short_strike, long_strike)
    
    for expiration in valid_expirations:
        for option_type in option_types:
            for delta in delta_targets:
                for width in spread_widths:
                    spread = find_spread_by_delta(
                        liquid_chain, option_type, expiration,
                        delta, width
                    )
                    
                    if spread is None:
                        continue
                    
                    if spread.credit_at_bid < min_credit:
                        continue
                    
                    # Deduplication key
                    key = (
                        spread.option_type,
                        spread.expiration_date,
                        spread.short_strike,
                        spread.long_strike
                    )
                    
                    if key in seen:
                        continue
                    
                    seen.add(key)
                    candidates.append(spread)
    
    return candidates


def print_chain_summary(chain: OptionChain) -> None:
    """Print summary of option chain."""
    print("\n" + "=" * 70)
    print("OPTION CHAIN SUMMARY")
    print("=" * 70)
    
    print(f"\nQuote Date: {chain.quote_date}")
    if chain.quote_time:
        print(f"Quote Time: {chain.quote_time}")
    print(f"Underlying: ${chain.underlying_price:.2f}")
    print(f"Total Contracts: {len(chain.contracts)}")
    print(f"  Puts: {len(chain.puts)}")
    print(f"  Calls: {len(chain.calls)}")
    
    print(f"\nExpirations ({len(chain.expirations)}):")
    for exp in chain.expirations[:5]:
        dte = (exp - chain.quote_date).days
        n_contracts = len([c for c in chain.contracts if c.expiration_date == exp])
        print(f"  {exp} ({dte} DTE): {n_contracts} contracts")
    if len(chain.expirations) > 5:
        print(f"  ... and {len(chain.expirations) - 5} more")
    
    print(f"\nStrike Range: ${min(chain.strikes):.0f} - ${max(chain.strikes):.0f}")
    
    # IV summary
    ivs = [c.implied_vol for c in chain.contracts]
    print(f"\nIV Range: {min(ivs):.1%} - {max(ivs):.1%}")
    print(f"IV Mean: {np.mean(ivs):.1%}")
    
    print("=" * 70)


def print_spread_candidates(spreads: List[MarketSpreadQuote]) -> None:
    """Print spread candidates table."""
    print("\n" + "=" * 100)
    print("MARKET SPREAD CANDIDATES")
    print("=" * 100)
    
    print(f"\n{'Type':<6} {'Exp':<12} {'Strikes':<12} {'DTE':<6} "
          f"{'Credit(bid)':<12} {'Credit(mid)':<12} {'MaxLoss':<10} {'Delta':<8}")
    print("-" * 100)
    
    for s in sorted(spreads, key=lambda x: (-x.credit_at_bid, x.dte)):
        type_str = "Put" if s.option_type == 'put' else "Call"
        strikes = f"{s.short_strike:.0f}/{s.long_strike:.0f}"
        
        print(f"{type_str:<6} {str(s.expiration_date):<12} {strikes:<12} {s.dte:<6} "
              f"${s.credit_at_bid:<11.2f} ${s.credit_at_mid:<11.2f} "
              f"${s.max_loss_at_bid:<9.2f} {s.net_delta:<7.2f}")
    
    print("=" * 100)
