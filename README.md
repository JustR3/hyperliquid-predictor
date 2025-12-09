# Cryptocurrency Price Predictor

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Active%20Development-yellow)

A machine learning-based trading strategy using XGBoost to predict cryptocurrency market movements with sophisticated risk management and realistic backtesting.

## Overview

This project uses XGBoost machine learning with the **Triple Barrier Method** for labeling to generate trading signals. The system features **multi-horizon prediction** (short/medium/long-term), combines technical indicators with BTC correlation, and uses volatility targeting for adaptive position sizing across multiple exchanges.

## 🎯 Key Features

### 1. **Multi-Horizon Prediction Modes**
Choose prediction timeframe based on your trading style:
- **Short-term (1-3 days)**: 3% profit / 1.5% loss targets - Quick trades
- **Medium-term (3-5 days)**: 5% profit / 2.5% loss targets - Swing trades (default)
- **Long-term (5-10 days)**: 8% profit / 4% loss targets - Position trades

Each mode uses adaptive Triple Barrier parameters for realistic labeling.

### 2. **Triple Barrier Method Labeling**
Instead of rigid binary thresholds, labels are created by observing which barrier is hit first:
- **Upper Barrier** (profit target): Label = 1 (Long win)
- **Lower Barrier** (stop loss): Label = -1 (Long loss)  
- **Time Barrier** (max holding): Label = sign of return (Neutral/timeout)

This captures realistic trade outcomes and is superior to arbitrary threshold classification.

### 3. **Advanced Features**
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, EMAs
- **BTC Beta**: Rolling correlation with BTC (market exposure)
- **Funding Rates**: Perpetual futures funding rates (sentiment indicator)
- **Volume Analysis**: Volume change, volume/SMA ratio
- **Price Momentum**: Multi-timeframe momentum (3d, 5d, 7d)

### 4. **Feature Importance Analysis**
Built-in tool to analyze which features contribute to predictions:
- Ranks all features by importance scores
- Identifies low-value features to remove
- Recommends optimal feature sets (top 80%, 90%, 95%)
- Mode-specific analysis for each prediction horizon

### 5. **Volatility Targeting**
Position sizing scales inversely with volatility to maintain constant risk:
- High volatility → Smaller positions
- Low volatility → Larger positions
- Target: 15% annualized portfolio volatility

### 6. **Realistic Cost Modeling**
- **Trading Fees**: 5 basis points (0.05%)
- **Slippage**: 10 basis points (0.1%)
- Combined impact on entry/exit prices

### 7. **Sharpe Ratio Optimization**
Hyperparameter tuning optimizes for **Sharpe Ratio** instead of accuracy, ensuring profitability matters more than prediction correctness.

### 8. **Multi-Exchange Support**
Works with 100+ exchanges via CCXT:
- Binance, Coinbase, Kraken, Bybit, OKX, Hyperliquid, etc.
- Symbol-specific hyperparameters with smart defaults

## 🚀 Installation

### Prerequisites
- Python 3.10+
- [UV](https://astral.sh/uv/) package manager (recommended) or pip
- macOS: `libomp` for XGBoost support

### Setup

```bash
# Clone repository
git clone https://github.com/JustR3/hyperliquid-predictor.git
cd hyperliquid-predictor

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .

# macOS only: Install OpenMP runtime
brew install libomp
```

## 📖 Usage

### Quick Start
```bash
# Run with defaults (BTC/USDT on Binance, medium-term mode)
python main.py
```

### Multi-Horizon Prediction
```bash
# Short-term (1-3 days) - Quick trades
python main.py --symbol BTC/USDT --mode short

# Medium-term (3-5 days) - Swing trades [DEFAULT]
python main.py --symbol BTC/USDT --mode medium

# Long-term (5-10 days) - Position trades
python main.py --symbol BTC/USDT --mode long
```

### Different Cryptocurrencies
```bash
# Bitcoin on Binance
python main.py --symbol BTC/USDT --exchange binance --mode short

# Ethereum on Coinbase  
python main.py --symbol ETH/USDT --exchange coinbase --mode medium

# Solana on Bybit
python main.py --symbol SOL/USDT --exchange bybit --mode long
```

### Hyperparameter Tuning (Optimizes for Sharpe Ratio)
```bash
# Optimize for specific symbol and mode
python tune.py --symbol BTC/USDT --mode short --trials 100 --save

# Use optimized parameters
python main.py --symbol BTC/USDT --mode short
```

### Feature Analysis
```bash
# Analyze which features are most important
python feature_analysis.py --symbol BTC/USDT --mode medium

# Compare across different coins
python feature_analysis.py --symbol ETH/USDC --mode long
python feature_analysis.py --symbol SOL/USDT --mode short
```

## 📁 Project Structure

```
hyperliquid-predictor/
├── config.py                 # Centralized configuration + prediction modes
├── main.py                   # Main orchestration with mode selection
├── tune.py                   # Optuna hyperparameter tuning (Sharpe Ratio)
├── feature_analysis.py       # Feature importance analysis tool
├── data/
│   ├── fetcher.py           # CCXT data fetching
│   ├── processor.py         # Feature engineering + Triple Barrier labels
│   └── storage.py           # Parquet caching
├── strategies/
│   └── xgb_strategy.py      # XGBoost model training/prediction
├── backtest/
│   └── engine.py            # Backtesting + RiskManager + volatility targeting
└── pyproject.toml           # Dependencies
```

**Why this structure?**
- **No `src/` folder**: Simpler imports
- **No base classes**: Just functions
- **Grouped by logic**: data/, strategies/, backtest/
- **Scalable**: Easy to add new models (e.g., `strategies/lgbm_strategy.py`)

## 🔬 Methodology

### Multi-Horizon Prediction
Three modes with adaptive Triple Barrier parameters:

| Mode | Horizon | Profit Target | Stop Loss | Risk:Reward |
|------|---------|---------------|-----------|-------------|
| Short | 3 days | 3% | 1.5% | 2:1 |
| Medium | 5 days | 5% | 2.5% | 2:1 |
| Long | 10 days | 8% | 4% | 2:1 |

### Triple Barrier Method
For each time `t`, we look forward and observe which happens first:
1. Price hits **upper barrier** (profit target) → Long win (Label = 1)
2. Price hits **lower barrier** (stop loss) → Long loss (Label = -1)
3. **Time barrier** passes without hitting targets → Timeout (Label = sign of return)

This creates more realistic labels than "will price go up tomorrow?"

### Position Sizing
Combines two methods:
1. **Volatility Targeting**: `size = (target_vol / current_vol) * capital`
2. **Kelly Criterion**: `size = kelly_fraction * (win_rate * avg_win - loss_rate * avg_loss)`
3. Take minimum of both for conservative sizing

### Backtest Types
1. **Standard**: Train once on 80% data, test on 20%
2. **Walk-Forward**: Retrain every 20 days on sliding 80-day window (more realistic)

## 📊 Output Example

```
==================================================
STANDARD BACKTEST
==================================================
Return: 23.45% | P&L: $2,345.67
Trades: 42 | Win Rate: 61.9%
Sharpe: 1.82 | Max DD: -8.34%

==================================================
WALK-FORWARD BACKTEST  
==================================================
Return: 18.23% | P&L: $1,823.12
Trades: 38 | Win Rate: 57.9%
Sharpe: 1.54 | Max DD: -11.21%

==================================================
TOP 5 FEATURES
==================================================
btc_beta: 0.1842
rsi: 0.1523
atr_pct: 0.1287
macd_histogram: 0.1156
funding_rate: 0.0982
==================================================
```

## 🔮 Roadmap / Future Enhancements

### ✅ Recently Completed
- [x] **Multi-Horizon Prediction Modes** - Short/medium/long-term with adaptive barriers
- [x] **Feature Importance Analysis** - Identify and optimize feature sets
- [x] **RiskManager Integration** - Consolidated risk management in backtest engine
- [x] **Walk-Forward Backtesting** - Realistic performance with periodic retraining

### 🎯 High Priority
- [ ] **Feature Set Optimization**
  - Remove zero-contribution features (funding_rate confirmed useless)
  - Test optimized feature sets via backtesting
  - Compare performance: All features vs Top 90% vs Top 80%

- [ ] **Probabilistic Targets (Quantile Regression)**
  - Predict P25, P50, P75 return distributions
  - Enter trades only when P75 > threshold (high confidence)
  - Captures uncertainty better than classification

### 🔧 Medium Priority
- [ ] **Alternative Models**
  - LightGBM (faster, better for time series)
  - Random Forest (simpler baseline)
  - Ensemble: Linear + Tree models

- [ ] **Cross-Asset Features**
  - SPY correlation (equity market sentiment)
  - DXY (dollar strength)
  - Gold correlation (risk-off indicator)

- [ ] **Advanced Indicators**
  - On-Balance Volume (OBV)
  - Ichimoku Cloud
  - ADX (trend strength)

### 🚀 Advanced
- [ ] **Online Learning**
  - Incremental model updates without full retraining
  - Libraries: Vowpal Wabbit, River

- [ ] **Multi-Timeframe Models**
  - Combine 1h + 1d data for better signals

- [ ] **Reinforcement Learning**
  - RL-based position sizing and timing

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- No guarantees of profitability
- Past performance ≠ future results  
- Cryptocurrency trading involves substantial risk of loss
- Only trade with capital you can afford to lose

Always do your own research and consult with financial advisors before trading.

---

**Star ⭐ this repo if you find it useful!**
