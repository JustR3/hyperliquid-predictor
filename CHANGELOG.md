# Changelog

## [2.0.0] - Multi-Symbol/Exchange Support (2025-01-15)

### Added
- **CLI Arguments for Symbol/Exchange Selection**
  - `--symbol`: Specify any trading pair (e.g., BTC/USDT, ETH/USDT, SOL/USDT)
  - `--exchange`: Choose from 100+ exchanges (binance, coinbase, kraken, bybit, okx, hyperliquid, etc.)
  - `--limit`: Customize historical data fetch limit (default: 200 days)

- **Symbol-Specific Hyperparameter Management**
  - Files named `best_hyperparameters_{SYMBOL}.json` (e.g., `best_hyperparameters_BTC_USDT.json`)
  - One config per symbol (not per exchange) - price behavior is asset-specific
  - Automatic loading with fallback hierarchy:
    1. Symbol-specific file (highest priority)
    2. Generic `best_hyperparameters.json` (fallback)
    3. Smart defaults based on volatility (last resort)

- **Smart Default Hyperparameters**
  - **Major caps** (BTC, ETH): Conservative params (shallow trees, low learning rate)
  - **Mid caps** (SOL, BNB, ADA, AVAX, DOT): Medium params
  - **Low caps**: Aggressive params (deep trees, higher learning rate)

- **Enhanced tune.py CLI**
  - Now supports `--symbol` and `--exchange` arguments
  - Saves symbol-specific config files
  - Provides clear workflow guidance

### Changed
- Refactored `main.py` to use `main()` function with argparse
- Updated `load_hyperparameters()` to accept symbol parameter
- Modified `walk_forward_backtest()` to pass symbol for hyperparameter loading
- All hardcoded "HYPE/USDT" and "hyperliquid" references now parameterized

### Technical Details
- Added `get_default_hyperparameters(symbol)` function with volatility-based defaults
- Exchange initialization now supports dynamic CCXT exchange loading
- Improved error handling for unsupported exchanges and missing symbols
- Symbol names sanitized for filesystem compatibility (/ → _)

### Usage Examples
```bash
# Run with defaults (HYPE/USDT on Hyperliquid)
uv run main.py

# Bitcoin on Binance
uv run main.py --symbol BTC/USDT --exchange binance

# Ethereum on Coinbase with 300 days of data
uv run main.py --symbol ETH/USDT --exchange coinbase --limit 300

# Tune hyperparameters for SOL
uv run tune.py --symbol SOL/USDT --exchange binance --trials 100 --save
```

### Migration Guide
**No breaking changes** - existing users can continue running `uv run main.py` without any modifications. The tool defaults to HYPE/USDT on Hyperliquid and will use existing `best_hyperparameters.json` if available.

To leverage new features:
1. Run hyperparameter tuning for your desired symbol:
   ```bash
   uv run tune.py --symbol YOUR_SYMBOL --exchange YOUR_EXCHANGE --trials 100 --save
   ```
2. Run main predictor with the same symbol:
   ```bash
   uv run main.py --symbol YOUR_SYMBOL --exchange YOUR_EXCHANGE
   ```

---

## [1.3.0] - Walk-Forward Testing (2025-01-10)

### Added
- Walk-forward backtesting with periodic model retraining
- Comparison between standard and walk-forward results
- More realistic performance evaluation

---

## [1.2.0] - Risk Management (2025-01-08)

### Added
- `risk_management.py` module with RiskManager class
- Kelly Criterion position sizing
- Stop-loss and take-profit automation
- Drawdown tracking and protection
- Comprehensive risk metrics display

---

## [1.1.0] - Technical Indicators Expansion (2025-01-05)

### Added
- MACD indicators (line, signal, histogram)
- EMA (9, 21, 50-period)
- Bollinger Bands
- Volume analysis (change, SMA ratio)
- Price momentum indicators

### Changed
- Expanded from 4 to 12 features
- Improved prediction accuracy (6.10% → 7.23% return)
- Profit factor increased (0.81x → 2.52x)

---

## [1.0.0] - Initial Release (2025-01-01)

### Added
- XGBoost classification model
- RSI and basic volume indicators
- Standard backtesting with comprehensive metrics
- Hyperparameter tuning via Optuna
- CCXT integration for Hyperliquid data fetching
