"""Shared evaluation utilities for terrain-classification notebooks.

Centralises the five helper functions that were previously copy-pasted across
07_baslines.ipynb, 07b_dataset_B_3class_baseline.ipynb, and 08_base_tuning.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError("xgboost is required. Install it before importing src.evaluation.") from exc


def make_base_models(random_state: int) -> dict[str, Pipeline]:
    """Return a dict of named sklearn Pipelines (scaler + classifier).

    Args:
        random_state: Seed passed to every stochastic estimator.

    Returns:
        ``{'LogisticRegression': ..., 'RandomForest': ..., 'XGBoost': ...}``
    """
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver="lbfgs",
                random_state=random_state,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=random_state,
            )),
        ]),
    }


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    evaluable_classes: list[str],
    label_order: list[str],
) -> dict[str, object]:
    """Compute per-fold accuracy, macro F1, and per-class metrics.

    Macro F1 is averaged only over *evaluable_classes* (classes present in both
    the training and test split).  Per-class metrics are set to NaN for any class
    absent from *evaluable_classes*.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        evaluable_classes: Classes to include in the macro-F1 average.
        label_order: Full class list used for the per-class breakdown.

    Returns:
        Dict with keys ``accuracy``, ``macro_f1``, ``per_class``, ``n_test``,
        ``degenerate_classes``.
    """
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)

    accuracy = accuracy_score(y_true, y_pred)

    if evaluable_classes:
        macro_f1 = f1_score(
            y_true, y_pred,
            labels=evaluable_classes,
            average="macro",
            zero_division=0,
        )
    else:
        macro_f1 = np.nan

    p_all, r_all, f_all, s_all = precision_recall_fscore_support(
        y_true, y_pred,
        labels=label_order,
        zero_division=0,
    )

    evaluable_set = set(evaluable_classes)
    per_class: dict = {}
    for idx, label in enumerate(label_order):
        if label in evaluable_set:
            per_class[label] = {
                "precision": float(p_all[idx]),
                "recall":    float(r_all[idx]),
                "f1":        float(f_all[idx]),
                "support":   int(s_all[idx]),
            }
        else:
            per_class[label] = {
                "precision": np.nan,
                "recall":    np.nan,
                "f1":        np.nan,
                "support":   int(s_all[idx]),
            }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1) if not np.isnan(macro_f1) else np.nan,
        "per_class": per_class,
        "n_test": int(len(y_true)),
        "degenerate_classes": [],
    }


def summarize_group(group: pd.DataFrame) -> pd.Series:
    """Aggregate per-fold metric rows into a summary Series.

    Computes mean, std (population), median, IQR, worst-fold value, and a
    *robustness score* (``macro_f1_mean - 0.5 * macro_f1_std``) for both
    accuracy and macro-F1.

    Args:
        group: DataFrame slice from a ``groupby`` containing at least
               ``accuracy`` and ``macro_f1`` columns.

    Returns:
        A pandas Series with summary statistics.
    """
    acc = group["accuracy"].dropna()
    mf1 = group["macro_f1"].dropna()

    def _q(s: pd.Series, q: float) -> float:
        return float(s.quantile(q)) if not s.empty else np.nan

    macro_f1_mean = float(mf1.mean()) if not mf1.empty else np.nan
    macro_f1_std  = float(mf1.std(ddof=0)) if not mf1.empty else np.nan
    robustness_score = (
        macro_f1_mean - 0.5 * macro_f1_std
        if not (np.isnan(macro_f1_mean) or np.isnan(macro_f1_std))
        else np.nan
    )

    return pd.Series({
        "accuracy_mean":       float(acc.mean())   if not acc.empty else np.nan,
        "accuracy_std":        float(acc.std(ddof=0)) if not acc.empty else np.nan,
        "accuracy_median":     float(acc.median()) if not acc.empty else np.nan,
        "accuracy_iqr":        _q(acc, 0.75) - _q(acc, 0.25) if not acc.empty else np.nan,
        "worst_fold_accuracy": float(acc.min())    if not acc.empty else np.nan,
        "macro_f1_mean":       macro_f1_mean,
        "macro_f1_std":        macro_f1_std,
        "macro_f1_median":     float(mf1.median()) if not mf1.empty else np.nan,
        "macro_f1_iqr":        _q(mf1, 0.75) - _q(mf1, 0.25) if not mf1.empty else np.nan,
        "worst_fold_macro_f1": float(mf1.min())    if not mf1.empty else np.nan,
        "robustness_score":    robustness_score,
        "evaluable_folds":     int(mf1.shape[0]),
    })


def row_normalize(cm: np.ndarray) -> np.ndarray:
    """Row-normalise a confusion matrix (divide each row by its sum).

    Rows with zero support are left as zeros.

    Args:
        cm: Raw integer confusion matrix of shape (n_classes, n_classes).

    Returns:
        Float array of the same shape with rows summing to 1 (or 0).
    """
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)


def fmt_mean_std(series: pd.Series) -> str:
    """Format a numeric Series as ``'mean +/- std'`` (3 decimal places).

    Returns ``'NaN'`` if the Series is empty after dropping NaNs.
    """
    vals = series.dropna()
    if vals.empty:
        return "NaN"
    return f"{vals.mean():.3f} +/- {vals.std(ddof=0):.3f}"
