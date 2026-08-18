"""
evaluator.py — Stratified K-Fold Evaluator & Threshold Optimizer
Computes Out-of-Fold (OOF) predictions, PR-AUC, ROC-AUC, and optimizes decision thresholds.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE


def evaluate_cv(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    use_smote: bool = False,
    smote_ratio: float = 0.5,
    early_stopping_rounds: int = 50,
) -> Dict[str, Any]:
    """
    Run Stratified K-Fold Cross Validation and record Out-of-Fold (OOF) predictions.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(y))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_val = X[val_idx], y[val_idx]

        if use_smote:
            sm = SMOTE(sampling_strategy=smote_ratio, random_state=random_state + fold)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

        model = model_factory()

        # Handle early stopping gracefully for different boosters
        try:
            if hasattr(model, "fit"):
                # XGBoost style
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_val)],
                    verbose=False,
                )
        except TypeError:
            model.fit(X_tr, y_tr)

        val_probs = model.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = val_probs

        fold_pr_auc = float(average_precision_score(y_val, val_probs))
        fold_roc_auc = float(roc_auc_score(y_val, val_probs))
        fold_scores.append({"fold": fold + 1, "pr_auc": fold_pr_auc, "roc_auc": fold_roc_auc})

    # Overall OOF Metrics
    oof_pr_auc = float(average_precision_score(y, oof_probs))
    oof_roc_auc = float(roc_auc_score(y, oof_probs))
    oof_logloss = float(log_loss(y, oof_probs))

    # Optimal Decision Threshold Search
    threshold_info = find_optimal_threshold(y, oof_probs)

    return {
        "oof_pr_auc": oof_pr_auc,
        "oof_roc_auc": oof_roc_auc,
        "oof_logloss": oof_logloss,
        "fold_scores": fold_scores,
        "optimal_threshold": threshold_info["optimal_threshold"],
        "precision": threshold_info["precision"],
        "recall": threshold_info["recall"],
        "f1_score": threshold_info["f1_score"],
        "confusion_matrix": threshold_info["confusion_matrix"],
        "oof_probs": oof_probs,
    }


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1", min_threshold: float = 0.15, max_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Scan probability thresholds to maximize F1-score or balance Precision vs Recall.
    """
    thresholds = np.linspace(min_threshold, max_threshold, 141)
    best_thresh = 0.5
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    best_cm = None

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            best_precision = float(p)
            best_recall = float(r)
            best_cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "optimal_threshold": round(best_thresh, 4),
        "f1_score": round(best_f1, 4),
        "precision": round(best_precision, 4),
        "recall": round(best_recall, 4),
        "confusion_matrix": best_cm,
    }
