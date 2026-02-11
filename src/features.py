from __future__ import annotations

"""Feature extraction utilities for windowed IMU/odometry signals."""

from typing import Sequence

import numpy as np
import pandas as pd


def _spectral_stats(signal: np.ndarray, fs_hz: float) -> tuple[float, float, float]:
    """Return simple frequency-domain descriptors for one 1D signal."""

    x = np.asarray(signal, dtype=float)
    x = x - np.nanmean(x)
    if len(x) < 4:
        return float("nan"), float("nan"), float("nan")

    fft_vals = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs_hz)
    power = np.abs(fft_vals) ** 2

    if len(power) <= 1:
        return float("nan"), float("nan"), float("nan")

    # Ignore DC (0 Hz) because we care about dynamic behavior.
    power_wo_dc = power[1:]
    freqs_wo_dc = freqs[1:]
    if np.all(power_wo_dc <= 0):
        return 0.0, 0.0, 0.0

    dominant_freq = float(freqs_wo_dc[np.argmax(power_wo_dc)])
    total_power = float(np.sum(power_wo_dc))

    p = power_wo_dc / total_power
    # Normalize entropy by log(N) so output is roughly in [0, 1].
    spectral_entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    return dominant_freq, total_power, spectral_entropy


def _time_stats(signal: np.ndarray) -> dict[str, float]:
    """Compute robust time-domain summary statistics for one signal."""

    x = np.asarray(signal, dtype=float)
    return {
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "median": float(np.nanmedian(x)),
        "iqr": float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
        "rms": float(np.sqrt(np.nanmean(x**2))),
    }


def compute_window_features(
    window: np.ndarray,
    channel_names: Sequence[str],
    fs_hz: float,
) -> dict[str, float]:
    """Extract feature dictionary for one window.

    Expected input shape:
    - rows: samples in time order
    - columns: sensor channels aligned with `channel_names`
    """

    if window.ndim != 2:
        raise ValueError("window must be 2D: [n_samples, n_channels]")
    if window.shape[1] != len(channel_names):
        raise ValueError("window channel count does not match channel_names length")

    feats: dict[str, float] = {}
    for i, col in enumerate(channel_names):
        sig = window[:, i]
        # Core descriptive stats.
        for k, v in _time_stats(sig).items():
            feats[f"{col}_{k}"] = v

        # Compact spectral signature.
        dom_f, power, entropy = _spectral_stats(sig, fs_hz=fs_hz)
        feats[f"{col}_dom_freq_hz"] = dom_f
        feats[f"{col}_spec_power"] = power
        feats[f"{col}_spec_entropy"] = entropy

        if len(sig) > 1:
            d1 = np.diff(sig)
            # First-difference features capture local roughness/jerkiness.
            feats[f"{col}_diff_std"] = float(np.nanstd(d1))
            feats[f"{col}_diff_rms"] = float(np.sqrt(np.nanmean(d1**2)))
        else:
            feats[f"{col}_diff_std"] = float("nan")
            feats[f"{col}_diff_rms"] = float("nan")

    return feats


def build_feature_table(
    window_tensor: np.ndarray,
    window_index: pd.DataFrame,
    channel_names: Sequence[str],
    fs_hz: float,
) -> pd.DataFrame:
    """Convert a full window tensor into one feature row per window."""

    if len(window_tensor) != len(window_index):
        raise ValueError("window_tensor and window_index must have the same number of windows")
    if len(window_tensor) == 0:
        # Keep metadata-only table when no valid windows survive filtering.
        return window_index.copy().reset_index(drop=True)

    rows = []
    for i in range(len(window_tensor)):
        row = compute_window_features(window_tensor[i], channel_names=channel_names, fs_hz=fs_hz)
        meta = window_index.iloc[i].to_dict()
        meta.update(row)
        rows.append(meta)

    return pd.DataFrame(rows)


def default_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return preferred channel order for feature extraction.

    Columns are returned only if present, so this works with partial schemas.
    """

    ordered = [
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "v1",
        "v2",
        "speed",
        "acc_mag",
        "gyro_mag",
        "wheel_diff_abs",
    ]
    return [c for c in ordered if c in df.columns]
