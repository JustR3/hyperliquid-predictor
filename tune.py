#!/usr/bin/env python3
"""
Hyperparameter tuning script for the XGBoost model.

This script optimizes model hyperparameters using Optuna and saves the best configuration.
Run this to find optimal hyperparameters before using them in main.py.

Usage:
    uv run tune.py --trials 100
"""

import argparse
import json
import ccxt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from hyperparameter_tuning import tune_hyperparameters, get_tuned_model, objective
from sklearn.metrics import accuracy_score


def compute_rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def main():
    parser = argparse.ArgumentParser(
        description='Tune XGBoost hyperparameters for HYPE/USDT prediction'
    )
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of optimization trials (default: 100)')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--save', action='store_true',
                        help='Save best parameters to file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("XGBoost Hyperparameter Tuning for HYPE/USDT")
    print("="*60)
    print()
    
    # Fetch data
    print(f"Fetching HYPE/USDT data from Hyperliquid...")
    exchange = ccxt.hyperliquid()
    ohlcv = exchange.fetch_ohlcv('HYPE/USDT', '1d', limit=200)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calculate features
    df['rsi'] = compute_rsi(df['close'], 14)
    df['vol_change'] = df['volume'].pct_change().fillna(0)
    df['price_change'] = df['close'].pct_change().shift(-1)
    
    # Manual inputs
    df['macro_score'] = 0.6
    df['unlock_pressure'] = 0.15
    
    # Target
    df['target'] = (df['close'].pct_change(7).shift(-7) > 0.05).astype(int)
    
    # Drop NaNs
    df = df.dropna()
    
    print(f"Data points: {len(df)}")
    print()
    
    # Prepare training data
    X = df[['rsi', 'vol_change', 'macro_score', 'unlock_pressure']]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print()
    
    # Run optimization
    print(f"Starting hyperparameter optimization ({args.trials} trials, {args.folds}-fold CV)...")
    print()
    
    best_params = tune_hyperparameters(
        X_train, y_train,
        n_trials=args.trials,
        n_splits=args.folds,
        verbose=True
    )
    
    print()
    print("="*60)
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    model = get_tuned_model(X_train, y_train, best_params)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print()
    
    # Save if requested
    if args.save:
        filename = 'best_hyperparameters.json'
        with open(filename, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"✓ Best parameters saved to {filename}")
        print()
        print("To use these parameters in main.py, update the model creation section:")
        print("  model = XGBClassifier(**best_params)")
        print()
    
    print("="*60)


if __name__ == '__main__':
    main()
