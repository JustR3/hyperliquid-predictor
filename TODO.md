# Project Roadmap

## ✅ Completed Features

### Core System
- [x] **Triple Barrier Method implementation** - Industry-standard labeling for realistic trade outcomes
- [x] **Multi-exchange support** - Works with 100+ exchanges via CCXT (Binance, Coinbase, Kraken, Bybit, OKX, Hyperliquid)
- [x] **XGBoost strategy** - Machine learning model with proper time-series handling
- [x] **Volatility targeting** - Dynamic position sizing based on market volatility
- [x] **Realistic cost modeling** - Trading fees (5 bps) + slippage (10 bps)
- [x] **Sharpe Ratio optimization** - Hyperparameter tuning focuses on profitability, not just accuracy
- [x] **Walk-forward backtesting** - Retrains model every 20 days to avoid overfitting
- [x] **Risk management system** - Kelly Criterion + volatility targeting for position sizing
- [x] **Feature engineering** - Technical indicators (RSI, MACD, BB, ATR, EMAs), BTC beta, funding rates
- [x] **Symbol-specific hyperparameters** - Smart defaults based on asset class (major/mid/low caps)
- [x] **Data caching** - Parquet storage for faster repeated analysis
- [x] **CLI interface** - Simple command-line interface with flexible options
- [x] **Type hints** - Full type annotation across all modules
- [x] **Centralized configuration** - All settings in `config.py`

### Documentation
- [x] **Comprehensive README** - Installation, usage, methodology, project structure
- [x] **MIT License** - Proper open-source licensing
- [x] **Code documentation** - Docstrings following Google style
- [x] **Project badges** - Python version, license, development status

---

## Agentic Workflow - Refactoring Plan

### Phase 1: Audit Data Layer
- [ ] **Verify `data/fetcher.py`**
  - [ ] Confirm async CCXT usage (or document why sync is acceptable)
  - [ ] Verify error handling for missing symbols
  - [ ] Test fallback mechanisms (USDT → USDC, primary exchange → Binance)
  - [ ] Validate rate limiting configuration

- [ ] **Verify `data/processor.py`**
  - [ ] Audit Triple Barrier Method implementation
  - [ ] Confirm profit/loss/time barrier logic is correct
  - [ ] Verify label remapping {-1, 0, 1} → {0, 1, 2} for XGBoost
  - [ ] Test edge cases (insufficient data, NaN handling)
  - [ ] Validate technical indicator calculations

### Phase 2: Audit Strategy Layer
- [ ] **Verify `strategies/xgb_strategy.py`**
  - [ ] Confirm hyperparameter loading hierarchy (symbol-specific → generic → smart defaults)
  - [ ] Verify label encoding/decoding matches training labels
  - [ ] Test model training with edge cases (single class, insufficient samples)
  - [ ] Validate feature importance extraction
  - [ ] Ensure prediction probabilities align with class labels

### Phase 3: Cleanup & Integration
- [ ] **Integrate `risk_management.py` into `backtest/engine.py`**
  - [ ] Move `RiskManager` class to `backtest/engine.py`
  - [ ] Move helper functions (`calculate_volatility`, `calculate_win_rate_and_ratio`)
  - [ ] Update imports in `main.py` and `tune.py`
  - [ ] Verify all tests still pass

- [ ] **Delete `risk_management.py`**
  - [ ] Confirm no remaining imports reference this file
  - [ ] Remove from repository

- [ ] **Organize `tune.py` output**
  - [ ] Ensure hyperparameters save to `data/best_hyperparameters_{SYMBOL}.json`
  - [ ] Verify file naming convention consistency
  - [ ] Add validation for saved JSON structure

## Future Enhancements (Post-Refactor)
- [ ] Implement probabilistic targets (quantile regression)
- [ ] Add multi-horizon predictions (1d, 3d, 7d)
- [ ] Explore alternative models (LightGBM, Random Forest)
- [ ] Add cross-asset features (SPY, DXY, Gold)
- [ ] Implement online learning for incremental updates

---

**Workflow**: For each task:
1. Read this roadmap
2. Plan changes
3. Execute code modifications
4. **VERIFY** by running the code
5. Commit with conventional commit message
