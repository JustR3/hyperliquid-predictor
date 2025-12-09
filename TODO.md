# Project Roadmap

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
