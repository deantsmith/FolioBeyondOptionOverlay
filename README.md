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
- **Exit Parameter Optimization**: Grid search over exit rules to maximize expected return
- **Kelly Criterion Position Sizing**: Optimal position sizing based on edge and risk
- **Risk Metrics**: Probability of Profit (POP), CVaR, risk-adjusted returns, and more
- **Live Portfolio Management**: Track positions, monitor exits, and manage trades

---

## What's New in This Branch

### Parameter Optimization (NEW)
- **`strategy/optimization.py`**: Complete optimization framework for exit parameters
- **`run_parameter_optimization.py`**: Command-line tool to optimize exit rules
- Grid search over profit targets, loss multiples, and DTE thresholds
- Maximize expected return subject to POP and CVaR constraints
- Outputs ranked parameter sets with performance metrics

### Enhanced Position Sizing
- **Kelly Criterion**: Automatic calculation of optimal position size
- Position sizing recommendations at multiple NAV levels ($1M, $10M)
- Portfolio NAV configuration in portfolio manager

### Bloomberg Integration Improvements
- Support for abbreviated option types ('c'/'p' → 'call'/'put')
- More robust CSV parsing

### Portfolio Manager Enhancements
- NAV setup on first run
- NAV management menu option
- Position sizing based on portfolio value

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
├── run_parameter_optimization.py  # Step 3: Optimize exit rule parameters (NEW)
├── portfolio_manager.py           # Step 4: Live portfolio management
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
│   ├── metrics.py                 # Risk metrics (POP, CVaR, Sortino, Kelly, etc.)
│   └── optimization.py            # Exit parameter optimization (NEW)
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

### Step 3: Optimize Exit Parameters (NEW)

Optimize exit rule parameters for a specific spread:
```bash
python run_parameter_optimization.py --calibration calibration_results.json \
    --delta 0.20 --width 3 --spread-type put --expiration 45
```

### Step 4: Manage Portfolio

Launch the interactive portfolio manager:
```bash
python portfolio_manager.py --data-dir ./portfolio_data
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
2024-12-30,15:45,91.50,p,2025-01-17,88,0.28,0.31,16.5,-0.15,0.038,-0.022,0.09,890,12100
2024-12-30,15:45,91.50,call,2025-01-17,94,0.35,0.38,15.8,0.25,0.042,-0.025,0.11,720,8450
```

| Field | Required | Format | Description |
|-------|----------|--------|-------------|
| `quote_date` | Yes | YYYY-MM-DD | Date of snapshot |
| `quote_time` | No | HH:MM | Time of snapshot |
| `underlying_price` | Yes | float | TLT spot price |
| `option_type` | Yes | put/call/p/c | Option type (lowercase, abbreviations supported) |
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

### Parameter Optimization (`run_parameter_optimization.py`) - NEW

Optimizes exit rule parameters for a specific spread strategy using grid search.

```bash
python run_parameter_optimization.py \
    --calibration calibration_results.json \
    --delta 0.20 \
    --width 3 \
    --spread-type put \
    --expiration 45 \
    --profit-targets 0.30,0.40,0.50,0.60,0.70 \
    --loss-multiples 1.5,2.0,2.5,3.0 \
    --dte-thresholds 3,5,7,10,14 \
    --min-pop 0.70 \
    --risk-budget 0.02375 \
    --output optimal_params.json
```

| Option | Default | Description |
|--------|---------|-------------|
| `--calibration` | Required | Path to calibration JSON |
| `--delta` | Required | Target delta for short strike (e.g., 0.20) |
| `--width` | Required | Spread width in dollars (e.g., 3.0) |
| `--spread-type` | Required | 'put' (bull put) or 'call' (bear call) |
| `--expiration` | 45 | Days to expiration |
| `--profit-targets` | 0.30,0.40,0.50,0.60,0.70 | Profit target %s to test |
| `--loss-multiples` | 1.5,2.0,2.5,3.0 | Loss multiples to test |
| `--dte-thresholds` | 3,5,7,10,14 | DTE thresholds to test |
| `--min-pop` | 0.70 | Minimum POP constraint |
| `--risk-budget` | None | Maximum CVaR loss budget |
| `--n-paths` | 10000 | Monte Carlo paths |
| `--seed` | 42 | Random seed |
| `--risk-free-rate` | 0.05 | Risk-free rate |
| `--output` | None | Path to save results JSON |
| `--top` | 10 | Number of top parameter sets to display |
| `--ranking-metric` | expected_return | Ranking metric ('expected_return' or 'risk_adjusted_return') |

#### Example Output

```
======================================================================
EXIT PARAMETER OPTIMIZATION
======================================================================

Spread: Put 88.00/85.00 (45 days)
Initial Credit: $0.72

Constraints:
  Min POP: 70%
  Max CVaR Loss: $2.38

Evaluating 100 parameter combinations...
  Profit targets: [0.3, 0.4, 0.5, 0.6, 0.7]
  Loss multiples: [1.5, 2.0, 2.5, 3.0]
  DTE thresholds: [3, 5, 7, 10, 14]

Optimization complete in 8.2s
  87/100 parameter sets pass constraints

======================================================================================
OPTIMIZATION RESULTS
======================================================================================

Best Parameters:
  Profit Target: 50%
  Loss Multiple: 2.5x
  DTE Threshold: 7 days

Performance:
  Expected Return: $0.52
  POP: 79.3%
  CVaR (95%): -$1.68
  Risk-Adj Return: 0.31
  Avg Days Held: 11.8

Exit Reason Distribution:
  profit_target        65.2%
  dte_threshold        23.5%
  loss_limit           7.8%
  expiration           3.5%


Top 10 Parameter Sets:
Rank   Profit%    LossMult   DTE      E[R]       POP      CVaR95     RAR
------------------------------------------------------------------------------------------
1      50%        2.5        7        $0.52      79.3%    $-1.68     0.31
2      50%        2.0        7        $0.51      78.9%    $-1.72     0.30
3      60%        2.5        7        $0.50      77.8%    $-1.65     0.30
4      40%        2.5        7        $0.50      79.8%    $-1.71     0.29
5      50%        2.5        10       $0.49      78.5%    $-1.70     0.29
...
```

---

## Exit Rules

The model implements three path-dependent exit conditions:

| Rule | Description | Default |
|------|-------------|---------|
| **Profit Target** | Close when unrealized P&L ≥ X% of max profit | 50% |
| **Loss Limit** | Close when spread value ≥ X× initial credit | 2.0× |
| **DTE Threshold** | Close N days before expiration | 7 days |

The first condition triggered determines the exit.

**Parameter Optimization** allows you to systematically search for optimal exit rules that maximize expected return subject to your risk constraints.

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
| **Kelly Fraction** | Optimal position size based on edge and odds (NEW) |

---

## Example Output

### Strategy Evaluation with Kelly Sizing (NEW)

```
======================================================================
TOP STRATEGIES (MARKET-PRICED)
======================================================================

Rank   Type       Strikes      Credit     POP      E[R]       CVaR95     RAR      Kelly%
----------------------------------------------------------------------------------------------------
1      Put        87/84        $0.72      78.5%    $0.48      -$1.52     0.32     15.8%
2      Put        86/83        $0.65      81.2%    $0.45      -$1.35     0.33     17.2%
3      Call       95/98        $0.58      76.8%    $0.41      -$1.42     0.29     14.5%

======================================================================

Top Strategy Details: Put 87/84 2025-02-14
  Initial Credit:     $0.72
  Max Profit:         $0.72
  Max Loss:           $2.28
  Avg Days Held:      12.3

  Position Sizing (Kelly):
    Kelly Fraction:   15.8%
    For $1M NAV:      52 contracts
    For $10M NAV:     526 contracts

  Exit Reason Distribution:
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

### Kelly Criterion Position Sizing (NEW)

The Kelly fraction determines optimal position size:

```
f* = (p × b - q) / b
```

Where:
- p = probability of profit
- q = 1 - p
- b = average win / average loss

The model computes Kelly fraction for each strategy and provides position sizing recommendations at different portfolio NAV levels.

**Conservative Approach**: Consider using a fractional Kelly (e.g., 0.5× or 0.25× Kelly) to reduce volatility.

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

# Exit Rules (defaults for evaluation)
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
│       - Includes Kelly position sizing recommendations          │
├─────────────────────────────────────────────────────────────────┤
│  3. PARAMETER OPTIMIZATION (run_parameter_optimization.py) NEW  │
│                                                                  │
│     Systematic search over exit rules:                          │
│       - Profit target percentages                               │
│       - Loss limit multiples                                    │
│       - DTE thresholds                                          │
│                                                                  │
│     Maximize expected return subject to:                        │
│       - Minimum POP constraint                                  │
│       - Maximum CVaR risk budget                                │
│                                                                  │
│     Output: Ranked parameter sets with metrics                  │
├─────────────────────────────────────────────────────────────────┤
│  4. PORTFOLIO MANAGEMENT (portfolio_manager.py)                 │
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
│       - portfolio_nav.json (NAV for position sizing) NEW        │
└─────────────────────────────────────────────────────────────────┘
```

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
- Setting portfolio NAV for position sizing (NEW)

### Data Files

The portfolio manager creates JSON files in your data directory:

| File | Description |
|------|-------------|
| `positions.json` | Open and closed positions with entry/exit details |
| `market_data.json` | Market snapshots and option price history |
| `portfolio_nav.json` | Portfolio NAV for position sizing (NEW) |

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
- Supports abbreviated option types ('c'/'p')

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
 11. Set portfolio NAV (NEW)
  Q. Quit

Enter choice: 1

================================================================================
PORTFOLIO SUMMARY
================================================================================

Portfolio NAV: $5,250,000

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

## Advanced Usage Examples

### Example 1: Complete Workflow

```bash
# 1. Calibrate models
python run_calibration.py --data historical_data.csv

# 2. Evaluate strategies with Bloomberg data
python run_market_evaluation.py \
    --calibration calibration_results.json \
    --options tlt_options_2025_01_09.csv \
    --min-pop 0.70 \
    --top 5

# 3. Optimize exit parameters for top strategy
python run_parameter_optimization.py \
    --calibration calibration_results.json \
    --delta 0.20 --width 3 --spread-type put \
    --min-pop 0.70 --risk-budget 0.02375 \
    --output optimal_exit_params.json

# 4. Launch portfolio manager
python portfolio_manager.py --data-dir ./portfolio_data
```

### Example 2: Parameter Sensitivity Analysis

Test different constraint levels:

```bash
# Conservative (higher POP requirement)
python run_parameter_optimization.py -c calibration.json \
    --delta 0.15 --width 3 --spread-type put \
    --min-pop 0.80 --risk-budget 0.01500

# Moderate
python run_parameter_optimization.py -c calibration.json \
    --delta 0.20 --width 3 --spread-type put \
    --min-pop 0.70 --risk-budget 0.02375

# Aggressive (maximize return)
python run_parameter_optimization.py -c calibration.json \
    --delta 0.25 --width 5 --spread-type put \
    --min-pop 0.60 --risk-budget 0.03000
```

### Example 3: Custom Parameter Ranges

Fine-tune the search grid:

```bash
python run_parameter_optimization.py -c calibration.json \
    --delta 0.20 --width 3 --spread-type put \
    --profit-targets 0.45,0.50,0.55,0.60,0.65 \
    --loss-multiples 1.8,2.0,2.2,2.4,2.6 \
    --dte-thresholds 5,6,7,8,9,10 \
    --min-pop 0.70
```

---

## Best Practices

### Strategy Selection
1. **Start with market evaluation**: Use Bloomberg data for realistic prices
2. **Apply constraints**: Set minimum POP (e.g., 70%) and CVaR budget
3. **Consider Kelly sizing**: Use Kelly fraction as a starting point, then apply fractional Kelly (0.25× - 0.5×) for conservative sizing
4. **Diversify**: Don't put all capital in a single strategy

### Exit Parameter Optimization
1. **Run on representative strategies**: Optimize for your typical delta and width
2. **Validate across market conditions**: Test parameters on different market environments
3. **Don't overfit**: Use broad parameter ranges, prefer simplicity
4. **Consider practical constraints**: Very tight stops may increase transaction costs

### Portfolio Management
1. **Daily monitoring**: Update market data and check exit signals daily
2. **Document trades**: Record entry/exit details for performance analysis
3. **Respect exit signals**: Follow the system's recommendations
4. **Review periodically**: Recalibrate models quarterly or when market regime changes

### Risk Management
1. **Position sizing**: Use Kelly criterion as guide, but apply fractional Kelly
2. **Risk budget**: Limit total portfolio CVaR to acceptable level
3. **Concentration limits**: Cap single position size (e.g., max 5% of NAV at risk)
4. **Stop-outs**: Honor loss limits and DTE thresholds

---

## Future Enhancements

Planned features for future releases:

- Rolling backtest engine for historical performance validation
- Multi-strategy portfolio optimization
- Greeks-based position hedging
- Automated Bloomberg data feed integration
- Performance analytics dashboard
- Alert/notification system for exit signals
- Machine learning for IV surface modeling
- Regime detection for dynamic parameter adjustment

---

## Troubleshooting

### Common Issues

**Issue**: Optimization finds no passing parameter sets
- **Solution**: Relax constraints (lower min_pop or increase risk_budget), or choose a different spread specification

**Issue**: Kelly fraction seems too high (>20%)
- **Solution**: Apply fractional Kelly (multiply by 0.25-0.5), Kelly assumes log utility and can be aggressive

**Issue**: CSV import fails with "Invalid option_type"
- **Solution**: Ensure option_type column contains 'put'/'call' or 'p'/'c' (lowercase)

**Issue**: Calibration R² is low (<0.95)
- **Solution**: Check data quality, consider different date range, or exclude volatile periods

**Issue**: Position sizing recommendations vary widely
- **Solution**: Kelly is sensitive to POP and payoff ratio, use as guideline not hard rule

---

## License

For internal use.

---

## Contact & Support

For questions, issues, or feature requests, please contact the development team.

**Repository**: (Add your GitHub URL here after pushing)

---

## Acknowledgments

This model builds on established quantitative finance methods:
- Vasicek (1977) interest rate model
- Black-Scholes-Merton option pricing framework
- Kelly (1956) criterion for optimal position sizing
- Rockafellar & Uryasev (2000) CVaR optimization

---

## Appendix: File Format Reference

### Calibration Output (`calibration_results.json`)

```json
{
  "vasicek_params": {
    "kappa_20": 0.15,
    "theta_20": 0.045,
    "sigma_20": 0.008,
    "kappa_30": 0.12,
    "theta_30": 0.048,
    "sigma_30": 0.009,
    "rho": 0.92
  },
  "regression_params": {
    "alpha": 5.2,
    "beta": -45.0,
    "gamma": 250.0,
    "r_squared": 0.987
  },
  "volatility_params": {
    "base_iv": 0.155,
    "price_sensitivity": -0.002
  },
  "current_values": {
    "yield_20y": 0.0438,
    "yield_30y": 0.0455,
    "tlt_price": 91.25,
    "date": "2025-01-09"
  }
}
```

### Optimization Output (`optimal_params.json`)

```json
{
  "spread": {
    "type": "bull_put",
    "short_strike": 88.0,
    "long_strike": 85.0,
    "expiration_days": 45
  },
  "initial_credit": 0.72,
  "best_params": {
    "profit_target_pct": 0.50,
    "loss_multiple": 2.5,
    "dte_threshold": 7
  },
  "best_result": {
    "expected_return": 0.52,
    "cvar_95": -1.68,
    "pop": 0.793,
    "risk_adjusted_return": 0.31,
    "avg_days_held": 11.8
  },
  "n_evaluated": 100,
  "n_passing": 87
}
```
