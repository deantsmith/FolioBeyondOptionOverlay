# TLT Options Overlay Model

A Python framework for modeling and optimizing credit spread strategies on TLT (iShares 20+ Year Treasury Bond ETF) as an income overlay for a fixed income portfolio.

## Overview

This model generates additional income from a fixed income portfolio by selling credit spreads on TLT. The approach is designed to limit losses to a fraction of the underlying portfolio yield (e.g., 50%), ensuring the overlay remains additive rather than destructive.

### Key Features

- **Interest Rate Modeling**: Correlated bivariate Vasicek process for 20-year and 30-year Treasury yields
- **TLT Price Simulation**: Regression-based conversion from yields to TLT prices with convexity adjustment
- **Monte Carlo Simulation**: 10,000+ path simulations for robust probability estimates
- **Path-Dependent Exit Rules**: Profit targets, loss limits, and DTE-based exits
- **Bloomberg Integration**: Use actual market IV and bid/ask prices for realistic evaluation
- **Risk Metrics**: Probability of Profit (POP), CVaR, risk-adjusted returns, and more

---

## Project Structure

```
tlt_options_model/
├── config.py                      # All parameters and settings
├── requirements.txt               # Python dependencies
│
├── run_calibration.py             # Step 1: Calibrate models from historical data
├── run_strategy_evaluation.py     # Step 2a: Evaluate strategies (model-based IV)
├── run_market_evaluation.py       # Step 2b: Evaluate strategies (Bloomberg IV)
├── portfolio_manager.py           # Step 3: Live portfolio management
├── test_with_synthetic_data.py    # Verify installation with synthetic data
│
├── data/
│   ├── loader.py                  # Historical yield/TLT data loading
│   └── bloomberg.py               # Bloomberg option chain integration
│
├── calibration/
│   ├── vasicek.py                 # Bivariate Vasicek MLE calibration
│   ├── tlt_regression.py          # TLT ~ yield regression with convexity
│   └── volatility.py              # Realized vol and IV-price relationship
│
├── simulation/
│   ├── rate_paths.py              # Monte Carlo yield path generation
│   └── tlt_paths.py               # Yield-to-TLT price conversion
│
├── pricing/
│   ├── black_scholes.py           # Black-Scholes pricing and Greeks
│   ├── spreads.py                 # Credit spread valuation
│   └── exits.py                   # Path-dependent exit logic
│
├── strategy/
│   ├── evaluation.py              # Strategy evaluation pipeline
│   └── metrics.py                 # Risk metrics (POP, CVaR, Sortino, etc.)
│
└── portfolio/
    ├── positions.py               # Position tracking and persistence
    ├── market_data.py             # Market data storage (yields, options)
    └── monitor.py                 # Position monitoring and exit signals
```

---

## Installation

### Requirements

- Python 3.8+
- NumPy, Pandas, SciPy

### Install Dependencies

```bash
pip install numpy pandas scipy
```

### Verify Installation

```bash
python test_with_synthetic_data.py
```

---

## Quick Start

### Step 1: Calibrate Models

Calibrate the interest rate and TLT models from historical data:

```bash
python run_calibration.py --data historical_data.csv --output calibration_results.json
```

### Step 2: Evaluate Strategies

**Option A**: Using model-estimated IV:
```bash
python run_strategy_evaluation.py --calibration calibration_results.json
```

**Option B**: Using Bloomberg market data:
```bash
python run_market_evaluation.py --calibration calibration_results.json --options tlt_options.csv
```

---

## Detailed Usage

### Calibration (`run_calibration.py`)

Calibrates three models from historical data:

1. **Bivariate Vasicek**: Mean-reverting process for 20Y and 30Y yields
2. **TLT Regression**: `log(TLT) = α + β×yield + γ×yield²`
3. **Volatility Model**: IV-price relationship for simulation

#### Historical Data CSV Format

```csv
date,yield_20y,yield_30y,tlt_close
2019-01-02,0.0285,0.0305,122.45
2019-01-03,0.0278,0.0298,124.12
```

| Column | Description |
|--------|-------------|
| `date` | Trading date (YYYY-MM-DD) |
| `yield_20y` | 20-year Treasury yield (decimal or percentage) |
| `yield_30y` | 30-year Treasury yield |
| `tlt_close` | TLT closing price |

#### Command Line Options

```bash
python run_calibration.py \
    --data historical_data.csv \
    --output calibration_results.json \
    --start-date 2019-01-01 \
    --end-date 2024-12-31 \
    --no-covid-exclusion
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | Required | Path to historical CSV |
| `--output` | `calibration_results.json` | Output file |
| `--start-date` | 2019-01-01 | Calibration start |
| `--end-date` | 2024-12-31 | Calibration end |
| `--no-covid-exclusion` | False | Include Mar 2020 - Dec 2021 |
| `--date-col` | `date` | Date column name |
| `--yield-20y-col` | `yield_20y` | 20Y yield column name |
| `--yield-30y-col` | `yield_30y` | 30Y yield column name |
| `--tlt-col` | `tlt_close` | TLT price column name |

---

### Strategy Evaluation (`run_strategy_evaluation.py`)

Evaluates credit spread strategies using Monte Carlo simulation with model-estimated IV.

```bash
python run_strategy_evaluation.py \
    --calibration calibration_results.json \
    --n-paths 10000 \
    --expiration 45 \
    --min-pop 0.70 \
    --profit-target 0.50 \
    --loss-multiple 2.0 \
    --dte-close 7 \
    --top 10
```

| Option | Default | Description |
|--------|---------|-------------|
| `--calibration` | Required | Path to calibration JSON |
| `--n-paths` | 10000 | Monte Carlo paths |
| `--expiration` | 45 | Days to expiration |
| `--min-pop` | 0.70 | Minimum probability of profit |
| `--profit-target` | 0.50 | Close at 50% of max profit |
| `--loss-multiple` | 2.0 | Close when spread value = 2× credit |
| `--dte-close` | 7 | Close 7 days before expiration |
| `--put-only` | False | Only evaluate put spreads |
| `--call-only` | False | Only evaluate call spreads |
| `--seed` | 42 | Random seed for reproducibility |
| `--top` | 10 | Number of top strategies to display |

---

### Market Evaluation (`run_market_evaluation.py`)

Evaluates strategies using actual Bloomberg option data for realistic pricing.

```bash
python run_market_evaluation.py \
    --calibration calibration_results.json \
    --options tlt_options.csv \
    --n-paths 10000 \
    --min-dte 30 \
    --max-dte 90 \
    --min-pop 0.70
```

| Option | Default | Description |
|--------|---------|-------------|
| `--calibration` | Required | Path to calibration JSON |
| `--options` | Required | Path to Bloomberg options CSV |
| `--n-paths` | 10000 | Monte Carlo paths |
| `--min-dte` | 30 | Minimum days to expiration |
| `--max-dte` | 90 | Maximum days to expiration |
| `--min-pop` | 0.70 | Minimum probability of profit |
| `--profit-target` | 0.50 | Profit target fraction |
| `--loss-multiple` | 2.0 | Loss limit multiple |
| `--dte-close` | 7 | DTE close threshold |

#### Bloomberg Option Chain CSV Format

```csv
quote_date,quote_time,underlying_price,option_type,expiration_date,strike,bid,ask,implied_vol,delta,gamma,theta,vega,volume,open_interest
2024-12-30,15:45,91.50,put,2025-01-17,89,0.42,0.45,16.2,-0.22,0.045,-0.028,0.12,1250,15420
2024-12-30,15:45,91.50,put,2025-01-17,88,0.28,0.31,16.5,-0.15,0.038,-0.022,0.09,890,12100
2024-12-30,15:45,91.50,call,2025-01-17,94,0.35,0.38,15.8,0.25,0.042,-0.025,0.11,720,8450
```

| Field | Required | Format | Description |
|-------|----------|--------|-------------|
| `quote_date` | Yes | YYYY-MM-DD | Date of snapshot |
| `quote_time` | No | HH:MM | Time of snapshot |
| `underlying_price` | Yes | float | TLT spot price |
| `option_type` | Yes | put/call | Option type (lowercase) |
| `expiration_date` | Yes | YYYY-MM-DD | Expiration date |
| `strike` | Yes | float | Strike price |
| `bid` | Yes | float | Bid price (dollars) |
| `ask` | Yes | float | Ask price (dollars) |
| `implied_vol` | Yes | float | IV as percentage (16.2 = 16.2%) |
| `delta` | Yes | float | Option delta |
| `gamma` | No | float | Gamma |
| `theta` | No | float | Theta (daily) |
| `vega` | No | float | Vega |
| `volume` | No | int | Daily volume |
| `open_interest` | No | int | Open interest |

---

## Exit Rules

The model implements three path-dependent exit conditions:

| Rule | Description | Default |
|------|-------------|---------|
| **Profit Target** | Close when unrealized P&L ≥ X% of max profit | 50% |
| **Loss Limit** | Close when spread value ≥ X× initial credit | 2.0× |
| **DTE Threshold** | Close N days before expiration | 7 days |

The first condition triggered determines the exit.

---

## Risk Metrics

For each strategy, the model computes:

| Metric | Description |
|--------|-------------|
| **POP** | Probability of Profit — % of paths with positive P&L |
| **Expected Return** | Mean P&L across all simulated paths |
| **CVaR (95%)** | Conditional Value at Risk — average loss in worst 5% of scenarios |
| **Risk-Adjusted Return** | Expected Return ÷ |CVaR| |
| **Profit Factor** | Gross profits ÷ Gross losses |
| **Win/Loss Ratio** | Average win ÷ Average loss |
| **Sortino Ratio** | Return ÷ Downside deviation |
| **Kelly Fraction** | Optimal position size based on edge and odds |

---

## Example Output

```
======================================================================
TOP STRATEGIES (MARKET-PRICED)
======================================================================

Rank   Type   Strikes      DTE    Credit     POP      E[R]       CVaR95     RAR     
----------------------------------------------------------------------------------------------
1      Put    87/84        45     $0.72      78.5%    $0.48      -$1.52     0.32    
2      Put    86/83        45     $0.65      81.2%    $0.45      -$1.35     0.33    
3      Call   95/98        52     $0.58      76.8%    $0.41      -$1.42     0.29    

======================================================================

Top Strategy Details:
  MarketSpreadQuote(Put 87/84 2025-02-14, credit=$0.68-$0.76)
  Credit at bid: $0.72
  Credit at mid: $0.74
  Max Loss: $2.28
  Net Delta: 0.145
  Net Theta: $0.018/day
  Avg Days Held: 12.3

  Exit Distribution:
    profit_target        62.5%
    dte_threshold        28.3%
    loss_limit            5.2%
    expiration            4.0%
```

---

## Model Details

### Vasicek Process

The model uses correlated Ornstein-Uhlenbeck (Vasicek) processes:

```
dr₂₀ = κ₂₀(θ₂₀ - r₂₀)dt + σ₂₀dW₂₀
dr₃₀ = κ₃₀(θ₃₀ - r₃₀)dt + σ₃₀dW₃₀

Corr(dW₂₀, dW₃₀) = ρ
```

**Parameters:**
- κ = mean reversion speed
- θ = long-run mean yield
- σ = volatility
- ρ = correlation between tenors

### TLT Regression

TLT price is modeled with a convexity adjustment:

```
log(TLT) = α + β × yield_avg + γ × yield_avg²
```

Where `yield_avg = (yield_20y + yield_30y) / 2`

The implied duration is approximately `-β` at the reference yield level.

### Credit Spread Types

| Type | Structure | Directional Bias |
|------|-----------|------------------|
| **Bull Put Spread** | Sell higher strike put, buy lower strike put | Neutral to bullish |
| **Bear Call Spread** | Sell lower strike call, buy higher strike call | Neutral to bearish |

Both are credit spreads with defined risk: **Max Loss = Spread Width - Credit Received**

---

## Configuration

Key parameters can be modified in `config.py`:

```python
# Risk Budget
portfolio_yield_annual = 0.0475   # 4.75% base portfolio yield
loss_budget_fraction = 0.50       # Max loss = 50% of yield = 2.375%

# Calibration
calibration_start = date(2019, 1, 1)
calibration_end = date(2024, 12, 31)
covid_start = date(2020, 3, 1)    # Excluded period
covid_end = date(2021, 12, 31)

# Simulation
n_paths = 10_000
trading_days_per_year = 252.0

# Exit Rules
profit_target_pct = 0.50
loss_limit_multiple = 2.0
dte_close_threshold = 7
```

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  1. CALIBRATION (run_calibration.py)                            │
│     Input: Historical yields + TLT prices                       │
│     Output: calibration_results.json                            │
├─────────────────────────────────────────────────────────────────┤
│  2. STRATEGY EVALUATION                                         │
│                                                                  │
│     Option A: Model-based (run_strategy_evaluation.py)          │
│       - Uses estimated IV from calibration                      │
│       - Good for initial screening                              │
│                                                                  │
│     Option B: Market-based (run_market_evaluation.py)           │
│       - Uses Bloomberg bid/ask and IV                           │
│       - Realistic entry prices                                  │
│       - Recommended for trade selection                         │
├─────────────────────────────────────────────────────────────────┤
│  3. PORTFOLIO MANAGEMENT (portfolio_manager.py)                 │
│                                                                  │
│     Daily workflow:                                              │
│       - Update market data (yields, TLT, option prices)         │
│       - Mark positions to market                                │
│       - Check exit signals                                      │
│       - Close positions / enter new trades                      │
│                                                                  │
│     Data files:                                                  │
│       - positions.json (trade history)                          │
│       - market_data.json (price history)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Future Enhancements

- Constrained optimization over exit rule parameters
- Rolling backtest on historical data
- Automated alerts/notifications when exit signals trigger
- Performance analytics and reporting

---

## Portfolio Management (Live Trading)

The portfolio management module operationalizes the model for live trading.

### Quick Start

```bash
python portfolio_manager.py --data-dir ./portfolio_data
```

This launches an interactive menu for:
- Adding new positions
- Updating market data (yields, TLT, option prices)
- Marking positions to market
- Checking exit signals
- Recording position closes

### Data Files

The portfolio manager creates two JSON files in your data directory:

| File | Description |
|------|-------------|
| `positions.json` | Open and closed positions with entry/exit details |
| `market_data.json` | Market snapshots and option price history |

### Workflow

**Daily Operations:**

1. **Update market data**
   - Enter current TLT price, 20Y yield, 30Y yield
   - Or import Bloomberg CSV with option prices

2. **Mark positions**
   - Calculates unrealized P&L for all open positions
   - Checks exit conditions (profit target, loss limit, DTE)

3. **Act on signals**
   - Close positions that hit exit triggers
   - Execute trades offline, record in system

4. **Enter new trades**
   - After running strategy evaluation, record new positions

### Position Entry

When entering a new position, you'll provide:

| Field | Description |
|-------|-------------|
| Option type | `put` or `call` |
| Short strike | Strike of the short leg |
| Long strike | Strike of the long leg |
| Expiration | Expiration date (YYYY-MM-DD) |
| Entry credit | Credit received per spread |
| Contracts | Number of spreads |
| TLT price | Underlying price at entry |
| IV (optional) | Implied volatility of short strike |
| Delta (optional) | Delta of short strike |

### Market Data Updates

**Option 1: Manual entry**
- Enter individual option prices through the menu

**Option 2: Bloomberg import**
- Import the same CSV format used for `run_market_evaluation.py`
- Automatically updates all option prices and TLT

### Exit Signals

The monitor checks three conditions (configurable):

| Signal | Default | Description |
|--------|---------|-------------|
| Profit Target | 50% | Close when P&L ≥ 50% of max profit |
| Loss Limit | 2.0x | Close when spread value ≥ 2× credit received |
| DTE Threshold | 7 days | Close when ≤ 7 days to expiration |

### Example Session

```
==================================================
TLT OPTIONS PORTFOLIO MANAGER
==================================================

  1. Dashboard (summary + alerts)
  2. Add new position
  3. Update market data (yields + TLT)
  4. Update option prices
  5. Mark positions to market
  6. Close position
  7. View all positions
  8. View market data
  9. Import Bloomberg CSV
 10. Configure exit rules
  Q. Quit

Enter choice: 1

================================================================================
PORTFOLIO SUMMARY
================================================================================

Market Data (2025-01-02T10:30):
  TLT: $91.25
  20Y: 4.380%  |  30Y: 4.550%

Positions:
  Open: 2
  Closed: 5

Open Position Totals:
  Total Credit Received: $720.00
  Total Max Loss: $2,280.00
  Unrealized P&L: +$285.00

⚠️  1 position(s) with exit signals!
```

### Programmatic Access

You can also use the portfolio modules directly in Python:

```python
from portfolio import (
    PositionStore, MarketDataStore, PositionMonitor,
    create_position, ExitRuleConfig
)

# Initialize stores
positions = PositionStore("positions.json")
market = MarketDataStore("market_data.json")

# Add a position
pos = create_position(
    option_type='put',
    short_strike=88,
    long_strike=85,
    expiration_date='2025-02-21',
    entry_credit=0.72,
    contracts=5,
    entry_underlying=91.50
)
positions.add_position(pos)

# Update market data
market.add_market_snapshot(tlt_price=91.25, yield_20y=4.38, yield_30y=4.55)
market.add_option_price('put', '2025-02-21', 88, bid=0.45, ask=0.52, implied_vol=16.8)
market.add_option_price('put', '2025-02-21', 85, bid=0.22, ask=0.28, implied_vol=17.5)

# Check positions
monitor = PositionMonitor(positions, market)
marks = monitor.mark_all_positions()

for mark in marks:
    print(f"{mark.position.position_id}: P&L ${mark.unrealized_pnl_mid:+.2f}")
    if mark.exit_signal.value != 'none':
        print(f"  ⚠️  {mark.exit_message}")
```

---

## License

For internal use.
