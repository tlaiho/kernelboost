"""Breast Cancer binary classification benchmark.

Compares KernelBoost against sklearn HistGradientBoostingClassifier,
XGBoost, and LightGBM on the Wisconsin Breast Cancer dataset.

Usage:
    python benchmarks/breast_cancer_benchmark.py [--no-gpu]
"""

import argparse
import time
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from kernelboost import KernelBooster
from kernelboost.feature_selection import SmartSelector
from kernelboost.objectives import EntropyObjective
from kernelboost.utilities import RankTransformer


def evaluate(model, X, y):
    """Return (accuracy, auc_roc, f1, predict_time) for a fitted model."""
    t0 = time.time()
    y_prob = np.asarray(model.predict_proba(X))
    pred_time = time.time() - t0
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    y_prob = y_prob.ravel()
    y_pred = (y_prob > 0.5).astype(int)
    return accuracy_score(y, y_pred), roc_auc_score(y, y_prob), f1_score(y, y_pred), pred_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breast Cancer benchmark")
    parser.add_argument("--no-gpu", action="store_true", help="CPU only")
    args = parser.parse_args()
    use_gpu = not args.no_gpu

    # Load and split data 
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split: train={len(X_train)}, test={len(X_test)}")

    # Preprocess 
    scaler = RankTransformer(pct=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #  KernelBoost 
    selector = SmartSelector(
        redundancy_penalty=0.4,
        relevance_alpha=0.7,
        recency_penalty=0.3,
        recency_decay=0.7,
        temperature=0.35,
        weight_decay=0.95,
    )

    kb = KernelBooster(
        objective=EntropyObjective(),
        max_sample=350,
        min_sample=80,
        n_estimators=100,
        subsample_share=0.85,
        lambda1=0.0002,
        learning_rate=0.85,
        min_features=1,
        max_features=5,
        overlap_epsilon=0.05,
        feature_selector=selector,
        use_gpu=use_gpu,
        verbose=0,
    )

    results = []

    t0 = time.time()
    kb.fit(X_train, y_train)
    kb_time = time.time() - t0
    acc, auc, f1, _ = evaluate(kb, X_test, y_test)
    results.append(("KernelBoost", acc, auc, f1, kb_time))

    # HGBC
    hgb = HistGradientBoostingClassifier(
        max_iter=150, max_depth=5, learning_rate=0.1,
    )

    t0 = time.time()
    hgb.fit(X_train, y_train)
    hgb_time = time.time() - t0
    acc, auc, f1, _ = evaluate(hgb, X_test, y_test)
    results.append(("sklearn HGBC", acc, auc, f1, hgb_time))

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
    )

    t0 = time.time()
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_time = time.time() - t0
    acc, auc, f1, _ = evaluate(xgb_model, X_test, y_test)
    results.append(("XGBoost", acc, auc, f1, xgb_time))

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1, verbose=-1,
    )

    t0 = time.time()
    lgb_model.fit(X_train, y_train)
    lgb_time = time.time() - t0
    acc, auc, f1, _ = evaluate(lgb_model, X_test, y_test)
    results.append(("LightGBM", acc, auc, f1, lgb_time))

    # --- Summary ---
    print("\n" + "=" * 65)
    print(f"{'Model':<20} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>10} {'Time':>10}")
    print("-" * 65)
    for name, acc, auc, f1, t in results:
        print(f"{name:<20} {acc:>10.4f} {auc:>10.4f} {f1:>10.4f} {t:>9.1f}s")
    print("=" * 65)
