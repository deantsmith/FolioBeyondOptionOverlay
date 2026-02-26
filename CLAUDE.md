# CLAUDE.md - Project Context for Claude Code

This file provides context for Claude Code when working on this project.

## Project Overview

**TLT Options Overlay Model** — A Python framework for generating additional income from a fixed income portfolio by selling credit spreads on TLT (iShares 20+ Year Treasury Bond ETF).

### Business Goal

A fixed income portfolio yielding ~4.75% annually wants to enhance returns through an options overlay strategy. The risk budget allows losses up to 50% of the base yield (2.375% annually), ensuring the overlay remains additive.

### Strategy

Sell credit spreads (bull put spreads and bear call spreads) on TLT with 30-90 DTE. Use Monte Carlo simulation to select strategies that maximize expected return subject to:
- Probability of Profit (POP) ≥ 70%
- CVaR (95%) within risk budget

---

## Architecture Decisions

### Why Model Interest Rates Instead of TLT Directly?

We model 20-year and 30-year Treasury yields using a bivariate Vasicek process, then convert to TLT prices via regression. This approach:
1. Leverages well-established rate models with analytical tractability
2. Captures the yield curve dynamics that drive TLT
3. Achieves R² ≈ 0.99 on the yield-to-TLT regression

### Why Vasicek Over CIR or Hull-White?

For 30-90 day option horizons:
- Mean reversion captures empirical yield behavior
- Closed-form transition density enables MLE calibration
- Additional complexity of CIR/Hull-White adds little value for short horizons
- Allows negative rates (less critical with COVID period excluded)

### COVID Period Exclusion

March 2020 - December 2021 is excluded from calibration by default. This period had atypical Fed intervention and rate dynamics that don't represent normal market behavior.

---

## Current State

### What's Built

| Module | Status | Description |
|--------|--------|-------------|
| `calibration/` | ✅ Complete | Bivariate Vasicek MLE, TLT regression, volatility estimation |
| `simulation/` | ✅ Complete | Monte Carlo yield paths, TLT price conversion |
| `pricing/` | ✅ Complete | Black-Scholes, credit spreads, path-dependent exits |
| `strategy/` | ✅ Complete | Strategy evaluation, risk metrics |
| `data/` | ✅ Complete | Historical data loader, Bloomberg integration |
| `portfolio/` | ✅ Complete | Position tracking, market data store, monitoring |

### Main Scripts

| Script | Purpose |
|--------|---------|
| `run_calibration.py` | Calibrate models from historical yield/TLT data |
| `run_strategy_evaluation.py` | Evaluate strategies with model-estimated IV |
| `run_market_evaluation.py` | Evaluate strategies with Bloomberg market data |
| `process_bloomberg_export.py` | Convert Bloomberg OMON exports to normalized options CSV |
| `portfolio_manager.py` | Interactive CLI for live portfolio management |
| `workbench_cli.py` | Unified text menu to run project routines and report created files |

### Data Files

| File | Format | Description |
|------|--------|-------------|
| Historical data | CSV | date, yield_20y, yield_30y, tlt_close |
| Bloomberg options | CSV | quote_date, underlying_price, option_type, expiration_date, strike, bid, ask, implied_vol, delta, ... |
| positions.json | JSON | Position history (entry/exit details) |
| market_data.json | JSON | Market snapshots and option prices |

---

## Key Design Details

### Exit Rules

Three path-dependent exit conditions (checked daily in simulation, configurable):

1. **Profit Target**: Close when unrealized P&L ≥ 50% of max profit
2. **Loss Limit**: Close when spread value ≥ 2× credit received
3. **DTE Threshold**: Close when ≤ 7 days to expiration

First condition triggered determines the exit.

### Risk Metrics

- **POP**: Probability of Profit (% of paths with positive P&L)
- **CVaR (95%)**: Average loss in worst 5% of scenarios
- **Risk-Adjusted Return**: Expected Return / |CVaR|

### Credit Spread Pricing

- Entry: Sell short leg at bid, buy long leg at ask (realistic fills)
- Mark-to-market: Use mid prices
- Exit: Buy back short at ask, sell long at bid (conservative)

---

## What's Not Yet Built

### Phase 4: Optimization (Planned)

- Systematic search over exit rule parameters
- Constrained optimization: maximize E[return] subject to CVaR ≤ budget
- Kelly criterion-based position sizing

### Other Enhancements (Future)

- Rolling backtest on historical data
- Automated alerts when exit signals trigger
- Performance analytics and reporting
- Greeks-based position adjustments

---

## Code Conventions

### File Organization

```
tlt_options_model/
├── calibration/     # Model calibration (Vasicek, regression, vol)
├── simulation/      # Monte Carlo path generation
├── pricing/         # Option pricing and spread valuation
├── strategy/        # Strategy evaluation and metrics
├── data/            # Data loading (historical + Bloomberg)
├── portfolio/       # Live position management
└── *.py             # Main scripts
```

### Data Classes

Most modules use `@dataclass` for structured data (e.g., `VasicekParams`, `SpreadPosition`, `OptionPriceUpdate`).

### Persistence

JSON files for simplicity and portability. Position and market data stores auto-save on modification.

### IV Format

- Bloomberg exports IV as percentage (16.2%)
- Internal storage uses decimal (0.162)
- Loaders auto-convert based on magnitude (>1.0 assumed percentage)

---

## Testing

```bash
# Verify installation with synthetic data
python test_with_synthetic_data.py

# Full pipeline test
python run_calibration.py --data historical_data.csv
python run_strategy_evaluation.py --calibration calibration_results.json
python run_market_evaluation.py --calibration calibration_results.json --options tlt_options.csv
```

---

## Common Tasks

### Run the Unified Workbench

```bash
python workbench_cli.py
```

The workbench presents a persistent text menu, runs selected routines with guided prompts, reports any file paths created during execution, and then returns to the menu until you quit.

### Add a New Risk Metric

1. Add function to `strategy/metrics.py`
2. Update `RiskMetrics` dataclass
3. Update `compute_all_metrics()` function

### Modify Exit Rules

1. Update `ExitRuleConfig` in `portfolio/monitor.py` or `pricing/exits.py`
2. Update `check_exit_conditions()` function
3. Add new `ExitSignal` enum value if needed

### Add New Data Source

1. Create loader in `data/` module
2. Convert to standard internal formats
3. Update `__init__.py` exports

---

## User Context

The user (Dean) works in portfolio management and quantitative finance with expertise in fixed income strategies and Bloomberg data integration. He is raising capital for fixed-income ETFs and building sophisticated quantitative frameworks. He prefers:
- Clean, well-documented code
- Comprehensive error handling
- Flexible command-line interfaces
- JSON for data persistence (simplicity over databases)

---

## Session History Summary

This project was built iteratively through conversation:

1. **Phase 1**: Designed and built calibration framework (Vasicek, TLT regression, volatility)
2. **Phase 2-3**: Built simulation engine and option pricing with exit logic
3. **Bloomberg Integration**: Added market data import and IV surface building
4. **Portfolio Management**: Built operational layer for live trading

The user has successfully:
- Run calibration on historical data
- Evaluated strategies with Bloomberg market data
- Tested the portfolio manager

Next likely steps:
- Push to GitHub (instructions provided)
- Continue daily operations
- Build Phase 4 optimization
