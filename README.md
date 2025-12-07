# Hyperliquid HYPE/USDT Price Predictor

A machine learning-based trading strategy for predicting HYPE/USDT price movements on the Hyperliquid exchange.

## Overview

This project uses XGBoost machine learning to predict whether HYPE/USDT will experience a >5% price increase over a 7-day period. The model combines technical indicators (RSI), volume analysis, and macro-sentiment inputs to generate trading signals.

## Features

- **Real-time Data Fetching**: Retrieves 200 days of OHLCV data from Hyperliquid via CCXT
- **Technical Indicators**: Calculates RSI (Relative Strength Index) for momentum analysis
- **XGBoost Classification**: Machine learning model trained on historical data
- **Comprehensive Backtesting**: Full P&L simulation with realistic trading mechanics
  - Position sizing (configurable % of capital)
  - Trading fees (basis points)
  - Equity curve tracking
  - Win rate, profit factor, Sharpe ratio
  - Maximum drawdown analysis
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

## Running the Application

```bash
uv run main.py
```

This will:
1. Fetch 200 days of HYPE/USDT data from Hyperliquid
2. Calculate technical indicators (RSI, volume change)
3. Train XGBoost model on 80% of historical data
4. Backtest on 20% of data and display performance metrics
5. Generate live prediction for next market move

## Output Example

```
==================================================
BACKTEST RESULTS
==================================================
Initial Capital: $10,000
Final Equity: $13,235.48
Total Return: 32.35%
Total P&L: $-58.37

Number of Trades: 3
Winning Trades: 1
Losing Trades: 2
Win Rate: 33.3%

Avg Win: $76.91
Avg Loss: $-67.64
Profit Factor: 0.57x

Sharpe Ratio: 6.29
Max Drawdown: -2.17%
==================================================

Probability of >5% move up in next 1-4 weeks: 17.5%
```

## Model Architecture

### Features
- **RSI (14-period)**: Relative Strength Index for momentum
- **Volume Change**: Percentage change in trading volume
- **Macro Score**: Manual sentiment score (0-1 scale)
- **Unlock Pressure**: Token unlock/dilution pressure estimate

### Target
Binary classification: 1 if next 7-day return > 5%, else 0

### XGBoost Hyperparameters
- Estimators: 200
- Max Depth: 4
- Learning Rate: 0.05
- Subsample: 0.8

## Backtest Parameters

- **Initial Capital**: $10,000
- **Position Size**: 10% of capital per trade
- **Trading Fee**: 5 basis points (0.05%)
- **Entry Signal**: Model prediction = 1
- **Exit Signal**: Model prediction = 0 or end of test period

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Total Return** | Overall profit/loss percentage |
| **Win Rate** | % of trades that were profitable |
| **Profit Factor** | Ratio of winning trades to losing trades |
| **Sharpe Ratio** | Risk-adjusted return (higher is better) |
| **Max Drawdown** | Largest peak-to-trough decline |

## Project Structure

```
hyperliquid-predictor/
├── main.py           # Main application
├── pyproject.toml    # Project dependencies (UV)
└── README.md         # This file
```

## Dependencies

- **ccxt**: Cryptocurrency exchange APIs
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **xgboost**: Gradient boosting machine learning
- **scikit-learn**: Machine learning utilities

## Disclaimer

This is an educational project for learning purposes. Cryptocurrency trading carries significant risk. Past performance does not guarantee future results. Never trade with capital you cannot afford to lose.

## Future Enhancements

- [ ] Real-time trading integration with Hyperliquid API
- [ ] Risk management features (stop-loss, take-profit)
- [ ] Additional technical indicators (MACD, Bollinger Bands)
- [ ] Hyperparameter optimization
- [ ] Walk-forward backtesting
- [ ] Multi-timeframe analysis
- [ ] Sentiment analysis integration
- [ ] Portfolio optimization across multiple assets

## License

MIT License

## Author

justra
