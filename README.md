# Cryptocurrency Price Predictor

A machine learning-based trading strategy for predicting cryptocurrency price movements across multiple exchanges.

## Overview

This project uses XGBoost machine learning to predict whether a cryptocurrency will experience a >5% price increase over a 7-day period. The model combines 12 technical indicators (RSI, MACD, EMA, Bollinger Bands, volume analysis), risk management, and macro-sentiment inputs to generate trading signals with comprehensive backtesting.

## Features

- **Multi-Symbol/Exchange Support**: Trade any cryptocurrency pair on 100+ exchanges (Binance, Coinbase, Kraken, Hyperliquid, etc.)
- **Symbol-Specific Hyperparameters**: Optimized parameters for each trading pair with smart defaults by asset class
- **12 Technical Indicators**: 
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - EMA (9, 21, 50-period Exponential Moving Averages)
  - Bollinger Bands
  - Volume analysis (change, SMA ratio)
  - Price momentum indicators
- **Advanced Risk Management**:
  - Kelly Criterion position sizing
  - Stop-loss and take-profit automation
  - Maximum drawdown limits
  - Capital preservation during adverse conditions
- **Dual Backtesting Modes**:
  - Standard: Train once on historical data
  - Walk-Forward: Periodic retraining every 20 days (more realistic)
- **Comprehensive Metrics**: P&L, win rate, profit factor, Sharpe ratio, max drawdown
- **Hyperparameter Optimization**: Bayesian optimization via Optuna
- **Real-time Predictions**: Live probability forecasts for upcoming market moves

## Installation

### Prerequisites
- Python 3.10+
- [UV](https://astral.sh/blog/uv/) package manager
- macOS: `libomp` for XGBoost support

### Setup

1. Clone the repository:
```bash
git clone https://github.com/justra/hyperliquid-predictor.git
cd hyperliquid-predictor
```

2. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

4. (macOS only) Install OpenMP runtime:
```bash
brew install libomp
```

## Usage

### Quick Start - Run with Defaults (HYPE/USDT on Hyperliquid)
```bash
uv run main.py
```

### Running with Different Cryptocurrencies

```bash
# Bitcoin on Binance
uv run main.py --symbol BTC/USDT --exchange binance

# Ethereum on Coinbase
uv run main.py --symbol ETH/USDT --exchange coinbase

# Solana on Bybit
uv run main.py --symbol SOL/USDT --exchange bybit

# Custom data limit (days of history)
uv run main.py --symbol BTC/USDT --exchange binance --limit 300
```

**Available exchanges**: binance, coinbase, kraken, bybit, okx, hyperliquid, and 100+ others via CCXT

### Hyperparameter Optimization

To optimize model parameters for a specific cryptocurrency:

```bash
# Quick tuning for BTC (20 trials, ~2 minutes)
uv run tune.py --symbol BTC/USDT --exchange binance --trials 20 --save

# Standard tuning for ETH (100 trials, ~10 minutes)
uv run tune.py --symbol ETH/USDT --exchange binance --trials 100 --save

# Comprehensive tuning for HYPE (400 trials, custom CV folds)
uv run tune.py --symbol HYPE/USDT --exchange hyperliquid --trials 400 --folds 5 --save
```

**How tuning works:**
- Uses Tree-structured Parzen Estimator (TPE) - a Bayesian optimization algorithm
- Saves results to `best_hyperparameters_{SYMBOL}.json` (e.g., `best_hyperparameters_BTC_USDT.json`)
- Parameters are automatically loaded when running `main.py` with the same symbol
- Optimizes: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma

**Hyperparameter strategy:**
- One config per symbol (not per exchange) since price behavior is asset-specific
- BTC/USDT on Binance and Kraken will use the same `best_hyperparameters_BTC_USDT.json`
- Falls back to generic `best_hyperparameters.json` if symbol-specific config doesn't exist
- Smart defaults based on asset volatility if no config files exist:
  - **Major caps** (BTC, ETH): Conservative parameters (shallow trees, low learning rate)
  - **Mid caps** (SOL, BNB, ADA): Medium parameters
  - **Low caps**: Aggressive parameters (deep trees, higher learning rate)

### Running the Main Predictor

```bash
uv run main.py --symbol BTC/USDT --exchange binance
```

This will:
1. Fetch 200 days of price data from the specified exchange
2. Calculate 12 technical indicators
3. Load optimized hyperparameters (symbol-specific > generic > smart defaults)
4. Train XGBoost model on 80% of historical data
5. Run two backtests:
   - **Standard**: Train once on historical data
   - **Walk-Forward**: Retrain every 20 days (more realistic)
6. Display comprehensive performance metrics
7. Generate live prediction probability

### Workflow for a New Cryptocurrency

1. **Tune hyperparameters** (optional but recommended):
```bash
uv run tune.py --symbol SOL/USDT --exchange binance --trials 100 --save
```

2. **Run the predictor**:
```bash
uv run main.py --symbol SOL/USDT --exchange binance
```

3. **Iterate**: Adjust risk parameters in `main.py` if needed:
   - `max_drawdown_pct`: Maximum allowed drawdown (default: 15%)
   - `max_position_size_pct`: Max position size (default: 20%)
   - `stop_loss_pct`: Stop-loss trigger (default: 3%)
   - `take_profit_pct`: Take-profit trigger (default: 8%)
   - `kelly_fraction`: Kelly criterion fraction (default: 30%)
2. Update the model creation in `main.py` with the optimized values
3. Re-run `main.py` with the new parameters

### Typical Workflow

```bash
# 1. Initial exploration
uv run main.py  # See current performance

# 2. Optimize model
uv run tune.py --trials 50 --save

# 3. Update main.py with best_hyperparameters.json values

# 4. Evaluate improved model
uv run main.py  # See improved backtest results
```

## Output Example

```
============================================================
CRYPTO PRICE PREDICTOR - BTC/USDT
============================================================
Exchange: binance
Historical Data: 200 days
============================================================

Fetching BTC/USDT data from binance...
✓ Fetched 200 days of data

✓ Loaded BTC/USDT hyperparameters from best_hyperparameters_BTC_USDT.json

==================================================
STANDARD BACKTEST (Train Once)
==================================================
Initial Capital: $10,000
Final Equity: $10,623.48
Total Return: 6.23%
Total P&L (net): $623.48
Total Fees Paid: $0.62

Number of Trades: 5
Winning Trades: 3
Losing Trades: 2
Win Rate: 60.0%

Avg Win: $241.56
Avg Loss: $-69.17
Profit Factor: 2.52x

Sharpe Ratio: 4.12
Max Drawdown: -1.23%
==================================================

==================================================
WALK-FORWARD BACKTEST (Retrain Every 20 Days)
==================================================
Initial Capital: $10,000
Final Equity: $10,723.19
Total Return: 7.23%
Total P&L (net): $723.19
Total Fees Paid: $0.72
Model Retrains: 5

Number of Trades: 6
Winning Trades: 4
Losing Trades: 2
Win Rate: 66.7%

Avg Win: $210.30
Avg Loss: $-65.40
Profit Factor: 2.64x

Sharpe Ratio: 4.89
Max Drawdown: -0.89%
==================================================

Probability of >5% move up in next 1-4 weeks: 42.1%
```

## Model Architecture

### Features (12 Total)
**Technical Indicators:**
- **RSI (14-period)**: Relative Strength Index for momentum
- **MACD**: Moving Average Convergence Divergence (line, signal, histogram)
- **Bollinger Bands**: Band position (where price sits within bands)
- **EMA Crossovers**: Price position relative to 9, 21, 50-period EMAs

**Volume Analysis:**
- **Volume Change**: Percentage change in trading volume
- **Volume/SMA Ratio**: Current volume vs 20-day average

**Momentum:**
- **5-Day Price Momentum**: Recent price change velocity

**Manual Inputs:**
- **Macro Score**: Market sentiment score (0-1 scale)
- **Unlock Pressure**: Token unlock/dilution pressure estimate

### Target
Binary classification: 1 if next 7-day return > 5%, else 0

### Risk Management
- **Kelly Criterion**: Optimal position sizing based on win rate and profit/loss ratio
- **Stop-Loss/Take-Profit**: Automatic exit at 3% loss or 8% gain
- **Drawdown Protection**: Trading halts if drawdown exceeds 15%
- **Position Limits**: Maximum 20% of capital per trade

## Backtest Explanation

The system runs **two backtests** to evaluate strategy performance:

### Standard Backtest (Train Once)
1. **Training (80%)**: Model learns patterns from 80% of historical data
2. **Testing (20%)**: Model makes predictions on unseen 20% of data
3. **Trade Simulation**: 
   - **Entry**: When model predicts 1 (expects >5% move up)
   - **Position Size**: Calculated via Kelly Criterion (capped at 20%)
   - **Fees**: 5 basis points on entry and exit
   - **Exit**: Stop-loss (3%), take-profit (8%), or model predicts 0

### Walk-Forward Backtest (More Realistic)
- Retrains model every 20 days with sliding 80-day window
- Simulates real-world periodic retraining
- Prevents look-ahead bias
- Results typically more conservative than standard backtest

**Metrics calculated:**
- **Total Return %**: Final equity vs initial capital
- **Total P&L (net)**: Profit/loss after fees (= Final Equity - Initial Capital)
- **Total Fees Paid**: Trading fees deducted
- **Win Rate**: % of profitable trades
- **Profit Factor**: Winning trade $ / Losing trade $ (>1 = profitable)
- **Sharpe Ratio**: Risk-adjusted returns (>1 is good, >2 is excellent)
- **Max Drawdown %**: Largest peak-to-trough decline

**Important notes:**
- Uses chronological split (not random) - tests on future data
- Includes realistic trading fees
- Walk-forward results are more trustworthy (less overfitting)
- Results show historical performance only - not guaranteed future results

## Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Total Return %** | Overall profit/loss as percentage of initial capital | >10% annually |
| **Total P&L (net)** | Actual dollar profit/loss after all fees | Positive |
| **Win Rate** | % of closed trades that were profitable | >50% |
| **Profit Factor** | Ratio of winning trades to losing trades | >1.5x |
| **Sharpe Ratio** | Risk-adjusted return | >2.0 |
| **Max Drawdown %** | Largest peak-to-trough decline | <20% |

## Project Structure

```
hyperliquid-predictor/
├── main.py                      # Main application (930 lines)
├── tune.py                      # Hyperparameter tuning CLI (191 lines)
├── hyperparameter_tuning.py     # Optuna optimization module (98 lines)
├── risk_management.py           # Risk management module (251 lines)
├── best_hyperparameters.json    # Generic optimized parameters
├── best_hyperparameters_*.json  # Symbol-specific optimized parameters
├── pyproject.toml               # Project dependencies (UV)
├── uv.lock                      # Locked dependency versions
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## File Descriptions

- **main.py** (930 lines)
  - Fetches OHLCV data from 100+ exchanges via CCXT
  - Computes 12 technical indicators (RSI, MACD, EMA, Bollinger Bands, volume)
  - Trains XGBoost model with symbol-specific optimized hyperparameters
  - Runs standard and walk-forward backtests with risk management
  - Generates real-time predictions
  - CLI arguments: --symbol, --exchange, --limit

- **tune.py** (191 lines)
  - Hyperparameter optimization using Optuna (Bayesian TPE algorithm)
  - Supports multi-symbol/exchange tuning
  - Saves symbol-specific config files
  - CLI arguments: --symbol, --exchange, --trials, --folds, --save

- **risk_management.py** (251 lines)
  - RiskManager class with Kelly Criterion position sizing
  - Stop-loss and take-profit automation
  - Drawdown tracking and protection
  - Position size limits and capital preservation

- **hyperparameter_tuning.py** (98 lines)
  - Reusable Optuna optimization module
  - Objective function for cross-validation scoring
  - Model training with optimized parameters
  - Reusable for future ML experiments

## Dependencies

- **ccxt** ≥4.0.0: Cryptocurrency exchange APIs (supports 100+ exchanges)
- **pandas** ≥2.0.0: Data manipulation and analysis
- **numpy** ≥1.24.0: Numerical computing
- **xgboost** ≥2.0.0: Gradient boosting machine learning
- **scikit-learn** ≥1.3.0: Cross-validation and metrics
- **optuna** ≥3.0.0: Bayesian hyperparameter optimization

## Disclaimer

This is an educational project for learning purposes. Cryptocurrency trading carries significant risk. Past performance does not guarantee future results. Never trade with capital you cannot afford to lose.

## Recent Updates

### v2.0 - Multi-Symbol/Exchange Support (Latest)
- ✅ Added CLI arguments for `--symbol` and `--exchange`
- ✅ Symbol-specific hyperparameter files (`best_hyperparameters_{SYMBOL}.json`)
- ✅ Smart default hyperparameters based on asset volatility
- ✅ Support for 100+ exchanges via CCXT
- ✅ Improved hyperparameter loading with fallback hierarchy

### v1.3 - Walk-Forward Testing
- ✅ Walk-forward backtesting with periodic retraining
- ✅ More realistic performance evaluation
- ✅ Comparison between standard and walk-forward results

### v1.2 - Risk Management
- ✅ Kelly Criterion position sizing
- ✅ Stop-loss and take-profit automation
- ✅ Drawdown tracking and protection
- ✅ Comprehensive risk metrics

### v1.1 - Technical Indicators Expansion
- ✅ Added MACD, EMA (9/21/50), Bollinger Bands
- ✅ Volume analysis (change, SMA ratio)
- ✅ Expanded from 4 to 12 features
- ✅ Improved prediction accuracy

### v1.0 - Initial Release
- ✅ XGBoost classification model
- ✅ RSI and basic volume indicators
- ✅ Standard backtesting
- ✅ Hyperparameter tuning via Optuna

## Future Enhancements

- [ ] Real-time trading integration with exchange APIs
- [ ] Automated position management via WebSockets
- [ ] Multi-timeframe analysis (1h, 4h, 1d combined)
- [ ] Ensemble methods combining multiple models
- [ ] Additional indicators (Volume Profile, Order Flow Imbalance)
- [ ] Portfolio-level risk management across multiple positions
- [ ] Alerts and notifications (email, Telegram)
- [ ] Web dashboard for monitoring and control
- [ ] Feature importance analysis and visualization
- [ ] Sentiment/macro data integration
- [ ] Trade history export (CSV/JSON)
- [ ] Equity curve visualization
- [ ] Monthly/weekly P&L reporting
- [ ] Model performance metrics (precision, recall, F1-score)
- [ ] Correlation analysis with other assets

## License

MIT License
