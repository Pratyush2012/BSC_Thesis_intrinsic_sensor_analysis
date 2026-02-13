from __future__ import annotations

"""Windowing utilities for converting continuous runs into fixed-size segments."""

from typing import Sequence

import numpy as np
import pandas as pd

from src.preprocess import infer_sampling_rate


def build_window_index(
    df: pd.DataFrame,
    window_s: float = 3.0,
    stride_s: float = 1.5,
    tcol: str = "t_rel",
    run_id: str | None = None,
    stationary_col: str = "is_stationary",
    max_stationary_ratio: float | None = 0.8,
    required_cols: Sequence[str] | None = None,
    min_valid_ratio: float = 0.95,
) -> pd.DataFrame:
    """Create metadata for overlapping fixed-length windows.

    This function does not copy sensor samples. It only records where each
    accepted window starts/ends, plus quality stats for filtering.
    """

    if window_s <= 0 or stride_s <= 0:
        raise ValueError("window_s and stride_s must be positive.")

    fs_hz = infer_sampling_rate(df, tcol=tcol)
    # Convert user-friendly seconds to exact sample counts on this dataset.
    window_n = max(1, int(round(window_s * fs_hz)))
    stride_n = max(1, int(round(stride_s * fs_hz)))
    n = len(df)

    if n < window_n:
        return pd.DataFrame(
            columns=[
                "window_id",
                "start_idx",
                "end_idx",
                "start_t",
                "end_t",
                "n_samples",
                "stationary_ratio",
                "valid_ratio",
                "run_id",
            ]
        )

    if required_cols is None:
        # Default to model-input signals (exclude metadata/flags).
        excluded = {"t", "t_rel", "run_id", "is_stationary", "is_stationary_raw"}
        required_cols = [c for c in df.columns if c not in excluded]

    rows = []
    win_id = 0
    for start in range(0, n - window_n + 1, stride_n):
        end = start + window_n
        seg = df.iloc[start:end]

        if stationary_col in seg.columns:
            # Useful for filtering out mostly-idle windows.
            stationary_ratio = float(seg[stationary_col].mean())
        else:
            stationary_ratio = float("nan")

        # Row is valid only if all required columns are present at that timestamp.
        valid_matrix = seg.loc[:, required_cols].notna().to_numpy(dtype=bool)
        valid_ratio = float(valid_matrix.all(axis=1).mean())

        if max_stationary_ratio is not None and not np.isnan(stationary_ratio):
            if stationary_ratio > max_stationary_ratio:
                continue
        if valid_ratio < min_valid_ratio:
            continue

        run_id_value = run_id
        if run_id_value is None and "run_id" in seg.columns:
            run_id_value = seg["run_id"].iloc[0]

        rows.append(
            {
                "window_id": win_id,
                "start_idx": int(start),
                "end_idx": int(end - 1),
                "start_t": float(seg[tcol].iloc[0]),
                "end_t": float(seg[tcol].iloc[-1]),
                "n_samples": int(window_n),
                "stationary_ratio": stationary_ratio,
                "valid_ratio": valid_ratio,
                "run_id": run_id_value,
            }
        )
        win_id += 1

    return pd.DataFrame(rows)


def extract_window_tensor(
    df: pd.DataFrame,
    window_index: pd.DataFrame,
    feature_cols: Sequence[str],
) -> np.ndarray:
    """Materialize window samples into a 3D tensor: [window, sample, channel]."""

    if window_index.empty:
        return np.empty((0, 0, len(feature_cols)), dtype=float)

    window_len = int(window_index["n_samples"].iloc[0])
    tensor = np.empty((len(window_index), window_len, len(feature_cols)), dtype=float)

    for i, row in enumerate(window_index.itertuples(index=False)):
        start = int(row.start_idx)
        end = int(row.end_idx) + 1
        window_values = df.iloc[start:end].loc[:, feature_cols].to_numpy(dtype=float)
        # Guard against accidental variable-length windows due to bad indices.
        if len(window_values) != window_len:
            raise ValueError("Variable window length found; expected fixed window size.")
        tensor[i] = window_values

    return tensor
