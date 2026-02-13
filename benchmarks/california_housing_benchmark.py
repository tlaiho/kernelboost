"""California Housing regression benchmark.

Compares KernelBoost against sklearn HistGradientBoostingRegressor,
XGBoost, and LightGBM on the California Housing dataset.

Usage:
    python benchmarks/california_housing_benchmark.py [--no-gpu]
"""

import argparse
import time
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from kernelboost import KernelBooster
from kernelboost.feature_selection import SmartSelector
from kernelboost.objectives import MSEObjective
from kernelboost.utilities import RankTransformer


def evaluate(model, X, y):
    """Return (mse, mae, r2, predict_time) for a fitted model."""
    t0 = time.time()
    y_pred = np.asarray(model.predict(X)).ravel()
    pred_time = time.time() - t0
    return mean_squared_error(y, y_pred), mean_absolute_error(y, y_pred), r2_score(y, y_pred), pred_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="California Housing benchmark")
    parser.add_argument("--no-gpu", action="store_true", help="CPU only")
    args = parser.parse_args()
    use_gpu = not args.no_gpu
    
    # Configuration 
    n_train = 16000
    n_val = 2400

    # Load and split data 
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    n_test = X.shape[0] - n_train - n_val
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=n_test, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=n_val, random_state=42)
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Preprocess 
    scaler = RankTransformer(pct=True)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # KernelBoost
    selector = SmartSelector(
        redundancy_penalty=0.4,
        relevance_alpha=0.7,
        recency_penalty=0.3,
        recency_decay=0.7,
        temperature=0.35,
        weight_decay=0.95,
        feature_groups=[[6, 7]],
    )

    kb = KernelBooster(
        objective=MSEObjective(),
        feature_selector=selector,
        max_sample=4000,
        min_sample=750,
        rounds=250,
        subsample_share=0.8,
        lambda1=0.0002, 
        learning_rate=0.8,
        min_features=1,
        max_features=5,
        overlap_epsilon=0.05,
        early_stopping_rounds=30,
        bounds=(0.10, 18.5),
        use_gpu=use_gpu,
        verbose=0,
    )

    results = []

    t0 = time.time()
    kb.fit(X_train, y_train, eval_set=(X_val, y_val))
    kb_time = time.time() - t0
    mse, mae, r2, _ = evaluate(kb, X_test, y_test)
    results.append(("KernelBoost", mse, mae, r2, kb_time))

    # HGBR  
    hgb = HistGradientBoostingRegressor(
        max_iter=250, max_depth=5, learning_rate=0.1,
    )

    t0 = time.time()
    hgb.fit(X_train, y_train)
    hgb_time = time.time() - t0
    mse, mae, r2, _ = evaluate(hgb, X_test, y_test)
    results.append(("sklearn HGBR", mse, mae, r2, hgb_time))

    # XGBoost  
    xgb_model = xgb.XGBRegressor(
        n_estimators=250, max_depth=5, learning_rate=0.1,
        early_stopping_rounds=30,
    )

    t0 = time.time()
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_time = time.time() - t0
    mse, mae, r2, _ = evaluate(xgb_model, X_test, y_test)
    results.append(("XGBoost", mse, mae, r2, xgb_time))

    #  LightGBM  
    lgb_model = lgb.LGBMRegressor(
        n_estimators=250, max_depth=5, learning_rate=0.1, verbose=-1,
    )

    t0 = time.time()
    lgb_model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    lgb_time = time.time() - t0
    mse, mae, r2, _ = evaluate(lgb_model, X_test, y_test)
    results.append(("LightGBM", mse, mae, r2, lgb_time))

    # Summary 
    print("\n" + "=" * 65)
    print(f"{'Model':<20} {'MSE':>10} {'MAE':>10} {'R²':>10} {'Time':>10}")
    print("-" * 65)
    for name, mse, mae, r2, t in results:
        print(f"{name:<20} {mse:>10.4f} {mae:>10.4f} {r2:>10.4f} {t:>9.1f}s")
    print("=" * 65)
