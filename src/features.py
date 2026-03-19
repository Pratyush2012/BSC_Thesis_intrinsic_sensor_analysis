from __future__ import annotations

"""Feature engineering and feature evaluation helpers.

This module centralizes reusable functionality previously implemented in notebooks:
- window-level feature extraction
- extended feature extraction
- velocity-based dataset splitting
- feature diagnostics and visualization (EDA)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ACC_COLS = ["ax", "ay", "az"]
GYRO_COLS = ["gx", "gy", "gz"]
IMU_COLS = ACC_COLS + GYRO_COLS
DEFAULT_ID_COLS = ["window_id", "run_id", "segment_id", "label", "t_start", "t_end", "n_samples"]
DEFAULT_BANDS = [
    (1.0, 5.0, "1_5Hz"),
    (5.0, 20.0, "5_20Hz"),
    (20.0, 40.0, "20_40Hz"),
    (40.0, 80.0, "40_80Hz"),
]


def _require_seaborn():
    try:
        return __import__("seaborn")
    except ImportError as exc:
        raise ImportError("Plotting functions require seaborn to be installed") from exc


def _as_float_array(x: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        return arr.ravel()
    return arr


def zero_crossing_rate(x: np.ndarray | pd.Series | list[float]) -> float:
    """Return normalized zero-crossing rate in [0, 1]."""

    arr = _as_float_array(x)
    if len(arr) < 2:
        return np.nan
    signs = np.sign(arr)
    # Treat exact zeros as previous sign to avoid artificial crossings.
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    crossings = np.sum(signs[1:] * signs[:-1] < 0)
    return float(crossings / (len(arr) - 1))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Robust Pearson correlation for two equal-length vectors."""

    if len(x) < 2 or len(y) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def band_power_fft(x: np.ndarray | pd.Series | list[float], fs: int, f_low: float, f_high: float) -> float:
    """Compute FFT-based band power in [f_low, f_high] Hz."""

    arr = _as_float_array(x)
    if len(arr) < 2:
        return np.nan

    arr_centered = arr - np.mean(arr)
    n = len(arr_centered)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(arr_centered)
    psd = (np.abs(fft_vals) ** 2) / n

    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def compute_fft_spectral_features(x: np.ndarray | pd.Series | list[float], fs: int) -> dict[str, float]:
    """Compute FFT-domain features: energy, dominant freq, centroid, entropy."""

    arr = _as_float_array(x)
    nan_result = {
        "spectral_energy": np.nan,
        "dominant_freq": np.nan,
        "spectral_centroid": np.nan,
        "spectral_entropy": np.nan,
    }
    if len(arr) < 2:
        return nan_result

    arr_centered = arr - np.mean(arr)
    n = len(arr_centered)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(arr_centered)
    psd = (np.abs(fft_vals) ** 2) / n

    total_power = float(np.trapezoid(psd, freqs))

    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    psd_pos = psd[pos_mask]

    if len(psd_pos) == 0 or np.sum(psd_pos) <= 0:
        return {
            "spectral_energy": total_power,
            "dominant_freq": np.nan,
            "spectral_centroid": np.nan,
            "spectral_entropy": np.nan,
        }

    dominant_freq = float(freqs_pos[np.argmax(psd_pos)])
    centroid = float(np.sum(freqs_pos * psd_pos) / np.sum(psd_pos))

    p = psd_pos / np.sum(psd_pos)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log2(p), 0.0)
    entropy = float(-np.sum(p * log_p))

    return {
        "spectral_energy": total_power,
        "dominant_freq": dominant_freq,
        "spectral_centroid": centroid,
        "spectral_entropy": entropy,
    }


def compute_time_features(x: np.ndarray | pd.Series | list[float]) -> dict[str, float]:
    """Compute statistical, shape, and energy time-domain features."""

    arr = _as_float_array(x)
    if len(arr) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "median": np.nan,
            "iqr": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "zero_crossing_rate": np.nan,
            "rms": np.nan,
            "sma": np.nan,
            "energy": np.nan,
            "ptp": np.nan,
            "mean_abs": np.nan,
        }

    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    std_v = float(np.std(arr))

    return {
        "mean": float(np.mean(arr)),
        "std": std_v,
        "min": min_v,
        "max": max_v,
        "range": max_v - min_v,
        "median": float(np.median(arr)),
        "iqr": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
        "skewness": float(sp_stats.skew(arr, bias=False)) if len(arr) >= 3 else np.nan,
        "kurtosis": float(sp_stats.kurtosis(arr, fisher=True, bias=False)) if len(arr) >= 4 else np.nan,
        "zero_crossing_rate": zero_crossing_rate(arr),
        "rms": float(np.sqrt(np.mean(arr ** 2))),
        "sma": float(np.mean(np.abs(arr))),
        "energy": float(np.sum(arr ** 2)),
        "ptp": float(np.ptp(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
    }


def compute_jerk_rms(x: np.ndarray | pd.Series | list[float], sample_rate: int = 100, scale_by_fs: bool = True) -> float:
    """Compute jerk RMS from first differences."""

    arr = _as_float_array(x)
    if len(arr) < 2:
        return np.nan
    jerk = np.diff(arr)
    if scale_by_fs:
        jerk = jerk * sample_rate
    return float(np.sqrt(np.mean(jerk ** 2)))


def _axis_feature_dict(
    sig: np.ndarray,
    fs: int,
    bands: list[tuple[float, float, str]],
) -> dict[str, float]:
    tf = compute_time_features(sig)
    sf = compute_fft_spectral_features(sig, fs)

    out = {
        **tf,
        **sf,
    }
    for f_low, f_high, suffix in bands:
        out[f"bandpower_{suffix}"] = band_power_fft(sig, fs, f_low, f_high)
    return out


def compute_window_features(
    windows_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    sample_rate: int = 100,
    bands: list[tuple[float, float, str]] | None = None,
    id_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute window-level feature table from raw signals."""

    if bands is None:
        bands = DEFAULT_BANDS
    if id_cols is None:
        id_cols = DEFAULT_ID_COLS

    required_signal_cols = ["run_id", "t_rel", "label"] + IMU_COLS
    missing_signal = [c for c in required_signal_cols if c not in signals_df.columns]
    if missing_signal:
        raise KeyError(f"Missing required columns in source dataset: {missing_signal}")

    required_window_cols = ["window_id", "run_id", "segment_id", "label", "t_start", "t_end"]
    missing_window = [c for c in required_window_cols if c not in windows_df.columns]
    if missing_window:
        raise KeyError(f"Missing required columns in window metadata: {missing_window}")

    feature_rows = []
    raw_acc_window_means = []
    norm_acc_window_means = []

    for _, w in windows_df.iterrows():
        run_id = w["run_id"]
        t_start = float(w["t_start"])
        t_end = float(w["t_end"])

        window_signal = signals_df[
            (signals_df["run_id"] == run_id)
            & (signals_df["t_rel"] >= t_start)
            & (signals_df["t_rel"] <= t_end)
        ]
        if window_signal.empty:
            continue

        acc_window_raw = window_signal[ACC_COLS].to_numpy(dtype=float)
        acc_window_centered = acc_window_raw - np.mean(acc_window_raw, axis=0, keepdims=True)
        acc_window_df = pd.DataFrame(acc_window_centered, columns=ACC_COLS, index=window_signal.index)

        raw_acc_window_means.append(np.mean(acc_window_raw, axis=0))
        norm_acc_window_means.append(np.mean(acc_window_centered, axis=0))

        row: dict[str, float | int | str] = {
            "window_id": int(w["window_id"]),
            "run_id": run_id,
            "segment_id": int(w["segment_id"]),
            "label": w["label"],
            "t_start": t_start,
            "t_end": t_end,
            "n_samples": int(len(window_signal)),
        }

        for col in IMU_COLS:
            if col in ACC_COLS:
                sig = acc_window_df[col].to_numpy(dtype=float)
            else:
                sig = window_signal[col].to_numpy(dtype=float)

            axis_feats = _axis_feature_dict(sig, sample_rate, bands)
            for k, v in axis_feats.items():
                row[f"{col}_{k}"] = v

            if col in ACC_COLS:
                # Backward compatibility with prior naming/definitions in 06_features.
                row[f"{col}_kurtosis"] = float(sp_stats.kurtosis(sig, fisher=True)) if len(sig) >= 2 else np.nan
                row[f"{col}_jerk_rms"] = compute_jerk_rms(sig, sample_rate=sample_rate, scale_by_fs=True)

        ay_detrended = acc_window_df["ay"].to_numpy(dtype=float)
        row["ay_detrended_rms"] = float(np.sqrt(np.mean(ay_detrended ** 2)))
        row["ay_detrended_std"] = float(np.std(ay_detrended))

        acc_mag = np.sqrt((acc_window_df ** 2).sum(axis=1).to_numpy(dtype=float))
        gyro_mag = np.sqrt((window_signal[GYRO_COLS] ** 2).sum(axis=1).to_numpy(dtype=float))

        for prefix, sig in [("acc_mag", acc_mag), ("gyro_mag", gyro_mag)]:
            mag_feats = _axis_feature_dict(sig, sample_rate, bands)
            for k, v in mag_feats.items():
                row[f"{prefix}_{k}"] = v

        acc_xyz = acc_window_df[ACC_COLS]
        row["acc_corr_xy"] = safe_corr(acc_xyz["ax"].to_numpy(), acc_xyz["ay"].to_numpy())
        row["acc_corr_xz"] = safe_corr(acc_xyz["ax"].to_numpy(), acc_xyz["az"].to_numpy())
        row["acc_corr_yz"] = safe_corr(acc_xyz["ay"].to_numpy(), acc_xyz["az"].to_numpy())

        gyro_xyz = window_signal[GYRO_COLS]
        row["gyro_corr_xy"] = safe_corr(gyro_xyz["gx"].to_numpy(dtype=float), gyro_xyz["gy"].to_numpy(dtype=float))
        row["gyro_corr_xz"] = safe_corr(gyro_xyz["gx"].to_numpy(dtype=float), gyro_xyz["gz"].to_numpy(dtype=float))
        row["gyro_corr_yz"] = safe_corr(gyro_xyz["gy"].to_numpy(dtype=float), gyro_xyz["gz"].to_numpy(dtype=float))

        feature_rows.append(row)

    features_df = pd.DataFrame(feature_rows).sort_values("window_id").reset_index(drop=True)
    if features_df.empty:
        raise RuntimeError("No features were generated. Check window metadata and source signals.")

    feature_cols = [c for c in features_df.columns if c not in id_cols]
    features_df = features_df[id_cols + feature_cols]

    if raw_acc_window_means:
        raw_means = np.vstack(raw_acc_window_means)
        norm_means = np.vstack(norm_acc_window_means)
        raw_abs_mean = np.mean(np.abs(raw_means), axis=0)
        norm_abs_mean = np.mean(np.abs(norm_means), axis=0)
        print("Mean absolute per-window accel mean (raw):")
        print(dict(zip(ACC_COLS, np.round(raw_abs_mean, 6))))
        print("Mean absolute per-window accel mean (normalized):")
        print(dict(zip(ACC_COLS, np.round(norm_abs_mean, 6))))

    return features_df


def band_power_welch(x: np.ndarray | pd.Series | list[float], fs: int, f_low: float, f_high: float) -> float:
    """Compute Welch-PSD band power in [f_low, f_high] Hz."""

    arr = _as_float_array(x)
    if len(arr) < 4:
        return np.nan
    nperseg = min(len(arr), 128)
    freqs, psd = sp_signal.welch(arr, fs=fs, nperseg=nperseg)
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def compute_spectral_features(x: np.ndarray | pd.Series | list[float], fs: int) -> dict[str, float]:
    """Compute spectral entropy, dominant frequency, and spectral centroid from Welch PSD."""

    nan_result = {
        "spectral_entropy": np.nan,
        "dominant_freq": np.nan,
        "spectral_centroid": np.nan,
    }
    arr = _as_float_array(x)
    if len(arr) < 4:
        return nan_result

    nperseg = min(len(arr), 128)
    freqs, psd = sp_signal.welch(arr, fs=fs, nperseg=nperseg)

    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    psd_pos = psd[pos_mask]
    total_pwr = psd_pos.sum()

    if total_pwr <= 0 or len(psd_pos) == 0:
        return nan_result

    p = psd_pos / total_pwr
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log2(p), 0.0)
    spectral_entropy = float(-np.sum(p * log_p))
    dominant_freq = float(freqs_pos[np.argmax(psd_pos)])
    spectral_centroid = float(np.sum(freqs_pos * psd_pos) / total_pwr)

    return {
        "spectral_entropy": spectral_entropy,
        "dominant_freq": dominant_freq,
        "spectral_centroid": spectral_centroid,
    }


def compute_hjorth_params(x: np.ndarray | pd.Series | list[float]) -> dict[str, float]:
    """Compute Hjorth activity, mobility, and complexity."""

    nan_result = {
        "hjorth_activity": np.nan,
        "hjorth_mobility": np.nan,
        "hjorth_complexity": np.nan,
    }
    arr = _as_float_array(x)
    if len(arr) < 3:
        return nan_result

    var_x = float(np.var(arr))
    std_x = np.sqrt(var_x)

    d1 = np.diff(arr)
    std_d1 = float(np.std(d1))

    activity = var_x
    if std_x < 1e-12:
        return {
            "hjorth_activity": activity,
            "hjorth_mobility": 0.0,
            "hjorth_complexity": 0.0,
        }

    mobility = std_d1 / std_x
    if len(d1) < 2 or std_d1 < 1e-12:
        complexity = 0.0
    else:
        d2 = np.diff(d1)
        std_d2 = float(np.std(d2))
        mobility_d1 = std_d2 / std_d1
        complexity = mobility_d1 / mobility if mobility > 1e-12 else 0.0

    return {
        "hjorth_activity": activity,
        "hjorth_mobility": float(mobility),
        "hjorth_complexity": float(complexity),
    }


def extract_extended_features(
    windows_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    fs: int = 100,
    vel_col: str = "net_speed",
) -> pd.DataFrame:
    """Extract extended time/frequency features for each window."""

    rows = []
    for _, w in windows_df.iterrows():
        run_id = w["run_id"]
        t_start = float(w["t_start"])
        t_end = float(w["t_end"])

        seg = signals_df[
            (signals_df["run_id"] == run_id)
            & (signals_df["t_rel"] >= t_start)
            & (signals_df["t_rel"] <= t_end)
        ]
        if seg.empty:
            continue

        row = {"window_id": int(w["window_id"])}

        acc_raw = seg[ACC_COLS].to_numpy(dtype=float)
        acc_centered = acc_raw - np.mean(acc_raw, axis=0, keepdims=True)
        acc_df = pd.DataFrame(acc_centered, columns=ACC_COLS)

        for col in ACC_COLS:
            sig = acc_df[col].to_numpy(dtype=float)
            n = len(sig)

            row[f"{col}_kurtosis"] = float(sp_stats.kurtosis(sig, fisher=True)) if n >= 2 else np.nan
            row[f"{col}_jerk_rms"] = float(np.sqrt(np.mean(np.diff(sig) ** 2))) if n >= 2 else np.nan

            sf = compute_spectral_features(sig, fs)
            row[f"{col}_spectral_entropy"] = sf["spectral_entropy"]
            row[f"{col}_dominant_freq"] = sf["dominant_freq"]
            row[f"{col}_spectral_centroid"] = sf["spectral_centroid"]
            row[f"{col}_bandpower_40_80Hz"] = band_power_welch(sig, fs, 40.0, 80.0)

            hp = compute_hjorth_params(sig)
            row[f"{col}_hjorth_activity"] = hp["hjorth_activity"]
            row[f"{col}_hjorth_mobility"] = hp["hjorth_mobility"]
            row[f"{col}_hjorth_complexity"] = hp["hjorth_complexity"]

        acc_mag = np.sqrt((acc_df ** 2).sum(axis=1).to_numpy())
        n_mag = len(acc_mag)

        row["acc_mag_kurtosis"] = float(sp_stats.kurtosis(acc_mag, fisher=True)) if n_mag >= 2 else np.nan
        row["acc_mag_jerk_rms"] = float(np.sqrt(np.mean(np.diff(acc_mag) ** 2))) if n_mag >= 2 else np.nan

        sf_mag = compute_spectral_features(acc_mag, fs)
        row["acc_mag_spectral_entropy"] = sf_mag["spectral_entropy"]
        row["acc_mag_dominant_freq"] = sf_mag["dominant_freq"]
        row["acc_mag_spectral_centroid"] = sf_mag["spectral_centroid"]
        row["acc_mag_bandpower_40_80Hz"] = band_power_welch(acc_mag, fs, 40.0, 80.0)

        hp_mag = compute_hjorth_params(acc_mag)
        row["acc_mag_hjorth_activity"] = hp_mag["hjorth_activity"]
        row["acc_mag_hjorth_mobility"] = hp_mag["hjorth_mobility"]
        row["acc_mag_hjorth_complexity"] = hp_mag["hjorth_complexity"]

        for col in GYRO_COLS:
            sig = seg[col].to_numpy(dtype=float)
            n = len(sig)

            row[f"{col}_kurtosis"] = float(sp_stats.kurtosis(sig, fisher=True)) if n >= 2 else np.nan
            row[f"{col}_jerk_rms"] = float(np.sqrt(np.mean(np.diff(sig) ** 2))) if n >= 2 else np.nan
            row[f"{col}_bandpower_40_80Hz"] = band_power_welch(sig, fs, 40.0, 80.0)

        gyro_mag = np.sqrt((seg[GYRO_COLS].to_numpy(dtype=float) ** 2).sum(axis=1))
        n_gmag = len(gyro_mag)

        row["gyro_mag_kurtosis"] = float(sp_stats.kurtosis(gyro_mag, fisher=True)) if n_gmag >= 2 else np.nan
        row["gyro_mag_jerk_rms"] = float(np.sqrt(np.mean(np.diff(gyro_mag) ** 2))) if n_gmag >= 2 else np.nan
        row["gyro_mag_bandpower_40_80Hz"] = band_power_welch(gyro_mag, fs, 40.0, 80.0)

        vel = seg[vel_col].to_numpy(dtype=float)
        row["odom_mean_velocity"] = float(np.mean(vel))
        row["odom_vel_std"] = float(np.std(vel))
        row["odom_distance"] = float(np.sum(np.abs(vel)) / fs)

        rows.append(row)

    return pd.DataFrame(rows)


def extract_new_features(
    windows_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    fs: int = 100,
    vel_col: str = "net_speed",
    label_col: str = "label",
) -> pd.DataFrame:
    """Compatibility wrapper for 06b naming."""

    _ = label_col
    return extract_extended_features(windows_df=windows_df, signals_df=signals_df, fs=fs, vel_col=vel_col)


def split_velocity_regime_datasets(
    df: pd.DataFrame,
    vel_col: str,
    mean_thresh: float = 0.5,
    std_thresh: float = 0.05,
    label_col: str = "label",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into full dataset (A) and low constant-velocity subset (B)."""

    if vel_col not in df.columns:
        raise KeyError(f"Column '{vel_col}' not found. Run extract_new_features() first.")
    if "odom_vel_std" not in df.columns:
        raise KeyError("Column 'odom_vel_std' not found. Run extract_new_features() first.")

    dataset_a = df.copy()
    mask = (df[vel_col] < mean_thresh) & (df["odom_vel_std"] < std_thresh)
    dataset_b = df[mask].copy().reset_index(drop=True)

    if verbose:
        n_total = len(df)
        n_kept = len(dataset_b)
        n_removed = n_total - n_kept

        print("=" * 60)
        print("DATASET A - Full Dataset")
        print("=" * 60)
        print(f"Total windows: {n_total}")
        print("Class distribution:")
        dist_a = dataset_a[label_col].value_counts().sort_index()
        for cls, cnt in dist_a.items():
            print(f"  {cls:<28} {cnt:>5}  ({cnt / n_total * 100:.1f}%)")

        print()
        print("=" * 60)
        print("DATASET B - Low Constant-Velocity Subset")
        print(f"  Criteria: {vel_col} < {mean_thresh} m/s  AND  odom_vel_std < {std_thresh} m/s")
        print("=" * 60)
        print(f"Windows kept:    {n_kept:>5}  ({n_kept / n_total * 100:.1f}%)")
        print(f"Windows removed: {n_removed:>5}  ({n_removed / n_total * 100:.1f}%)")
        print("Class distribution:")
        dist_b = dataset_b[label_col].value_counts().sort_index()
        for cls, cnt in dist_b.items():
            pct = cnt / n_kept * 100 if n_kept > 0 else 0.0
            print(f"  {cls:<28} {cnt:>5}  ({pct:.1f}%)")

    return dataset_a, dataset_b


def create_velocity_filtered_dataset(
    df: pd.DataFrame,
    vel_col: str,
    mean_thresh: float = 0.5,
    std_thresh: float = 0.05,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper for 06b naming."""

    return split_velocity_regime_datasets(
        df=df,
        vel_col=vel_col,
        mean_thresh=mean_thresh,
        std_thresh=std_thresh,
        label_col=label_col,
        verbose=True,
    )


def get_feature_columns(features_df: pd.DataFrame, id_cols: list[str] | None = None) -> list[str]:
    """Return columns treated as model features."""

    if id_cols is None:
        id_cols = DEFAULT_ID_COLS
    return [c for c in features_df.columns if c not in id_cols]


def get_legacy_feature_columns(features_df: pd.DataFrame, id_cols: list[str] | None = None) -> list[str]:
    """Return a pre-expansion style feature subset for backward-compatible EDA."""

    base = get_feature_columns(features_df, id_cols=id_cols)
    exclude_tokens = [
        "_min",
        "_max",
        "_range",
        "_median",
        "_iqr",
        "_skewness",
        "_zero_crossing_rate",
        "_sma",
        "_spectral_energy",
        "_dominant_freq",
        "_spectral_centroid",
        "_spectral_entropy",
        "_bandpower_40_80Hz",
        "acc_corr_",
        "gyro_corr_",
    ]
    return [c for c in base if not any(tok in c for tok in exclude_tokens)]


def print_feature_diagnostics(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    label_col: str = "label",
) -> pd.DataFrame:
    """Print diagnostics and return label summary DataFrame."""

    if feature_cols is None:
        feature_cols = get_feature_columns(features_df)

    print("Feature table diagnostics")
    print("=" * 80)
    print(f"Shape: {features_df.shape}")
    print(f"Feature count: {len(feature_cols)}")

    missing_counts = features_df[feature_cols].isna().sum()
    missing_total = int(missing_counts.sum())
    print(f"Total missing values in feature columns: {missing_total}")
    if missing_total > 0:
        print("Top columns with missing values:")
        print(missing_counts[missing_counts > 0].sort_values(ascending=False).head(10).to_string())

    label_counts = features_df[label_col].value_counts().sort_index()
    label_pct = (label_counts / len(features_df) * 100).round(2)
    label_summary = pd.DataFrame({"count": label_counts, "pct": label_pct})
    print("\nLabel distribution:")
    print(label_summary.to_string())

    key_check_cols = [
        "acc_mag_rms",
        "acc_mag_std",
        "acc_mag_energy",
        "gyro_mag_rms",
        "gyro_mag_std",
        "gyro_mag_energy",
    ]
    key_check_cols = [c for c in key_check_cols if c in features_df.columns]
    if key_check_cols:
        print("\nQuick summary (key separator candidates):")
        print(features_df.groupby(label_col)[key_check_cols].mean().round(4).to_string())

    return label_summary


def plot_feature_distributions(
    features_df: pd.DataFrame,
    viz_features: list[str],
    reports_dir: str | Path,
    label_col: str = "label",
    show: bool = True,
) -> list[Path]:
    """Plot and save histogram+violin distributions for selected features."""

    sns = _require_seaborn()

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    viz_features = [c for c in viz_features if c in features_df.columns]
    if not viz_features:
        raise RuntimeError("No requested visualization features were found in features_df.")

    out_files: list[Path] = []
    for feat in viz_features:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        #sns.histplot(
        #    data=features_df,
        #    x=feat,
        #    hue=label_col,
        #    bins=30,
        #    stat="count",
        #    common_norm=False,
        #    alpha=0.5,
        #    edgecolor="black",
        #    linewidth=0.8,
        #    ax=axes[0],
        #)

        axes[0].hist(
            [features_df[features_df[label_col] == cls][feat].dropna() for cls in features_df[label_col].unique()],
            bins=25,
            alpha=0.5,
            edgecolor="black",
            label=[str(cls) for cls in features_df[label_col].unique()],
        )

        axes[0].set_title(f"{feat} distribution by label")
        axes[0].set_xlabel(feat)
        axes[0].set_ylabel("count")
        axes[0].legend()

        sns.violinplot(data=features_df, x=label_col, y=feat, ax=axes[1], inner="quartile", cut=0)
        axes[1].set_title(f"{feat} violin by label")
        axes[1].tick_params(axis="x", rotation=20)

        fig.tight_layout()
        out_path = reports_path / f"dist_{feat}.png"
        fig.savefig(out_path, bbox_inches="tight")
        out_files.append(out_path)
        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"Saved {len(viz_features)} distribution figures to {reports_path}")
    return out_files


def analyze_feature_correlations(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    reports_dir: str | Path | None = None,
    top_n_heatmap: int = 20,
    show: bool = True,
) -> pd.DataFrame:
    """Create correlation heatmap and return top absolute correlation pairs."""

    sns = _require_seaborn()

    if feature_cols is None:
        feature_cols = get_feature_columns(features_df)

    feature_std = features_df[feature_cols].std(numeric_only=True)
    ranked = feature_std.sort_values(ascending=False)
    heatmap_cols = ranked.head(min(top_n_heatmap, len(ranked))).index.tolist()

    corr = features_df[heatmap_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title(f"Correlation heatmap (top {len(heatmap_cols)} variable features)")
    fig.tight_layout()

    if reports_dir is not None:
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(reports_path / "corr_heatmap.png", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            pairs.append((cols[i], cols[j], val, abs(val)))

    pairs_df = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "corr", "abs_corr"])
    high_corr = pairs_df.sort_values("abs_corr", ascending=False).head(15)

    print("Top 15 absolute correlations among heatmap features:")
    print(high_corr[["feature_1", "feature_2", "corr"]].to_string(index=False))
    return high_corr


def _stratified_sample_for_tsne(
    features_df: pd.DataFrame,
    label_col: str,
    sample_cap: int,
    random_state: int,
) -> pd.DataFrame:
    if len(features_df) <= sample_cap:
        return features_df

    sampled = features_df.groupby(label_col, group_keys=False).apply(
        lambda g: g.sample(
            n=max(1, int(sample_cap * len(g) / len(features_df))),
            random_state=random_state,
        )
    )
    return sampled.reset_index(drop=True)


def plot_dimensionality_reduction(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    label_col: str = "label",
    reports_dir: str | Path | None = None,
    random_state: int = 42,
    tsne_sample_cap: int = 2500,
    show: bool = True,
) -> dict[str, np.ndarray | float]:
    """Plot PCA and t-SNE projections and save figures if requested."""

    sns = _require_seaborn()

    if feature_cols is None:
        feature_cols = get_feature_columns(features_df)

    reports_path = None
    if reports_dir is not None:
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)

    X = features_df[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        label_col: features_df[label_col].values,
    })

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=label_col, alpha=0.7, s=40, ax=ax)
    ax.set_title("PCA (2D) by label")
    ax.legend(title=label_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    if reports_path is not None:
        fig.savefig(reports_path / "pca_2d.png", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    print("PCA cumulative explained variance:", pca.explained_variance_ratio_.sum())

    sampled_df = _stratified_sample_for_tsne(
        features_df=features_df,
        label_col=label_col,
        sample_cap=tsne_sample_cap,
        random_state=random_state,
    )
    tsne_input = sampled_df[feature_cols].to_numpy(dtype=float)
    tsne_input = np.nan_to_num(tsne_input, nan=0.0, posinf=0.0, neginf=0.0)
    tsne_input = scaler.fit_transform(tsne_input)
    tsne_labels = sampled_df[label_col].to_numpy()

    perplexity = min(30, max(5, len(tsne_input) - 1))
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    X_tsne = tsne.fit_transform(tsne_input)

    tsne_df = pd.DataFrame({
        "TSNE1": X_tsne[:, 0],
        "TSNE2": X_tsne[:, 1],
        label_col: tsne_labels,
    })

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue=label_col, alpha=0.75, s=40, ax=ax)
    ax.set_title("t-SNE (2D) by label")
    ax.legend(title=label_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    if reports_path is not None:
        fig.savefig(reports_path / "tsne_2d.png", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    if reports_path is not None:
        print(f"Saved PCA and t-SNE figures to: {reports_path}")

    return {
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "pca_cumulative_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "n_tsne_samples": int(len(tsne_input)),
    }


def run_feature_evaluation(
    features_df: pd.DataFrame,
    reports_dir: str | Path,
    viz_features: list[str],
    label_col: str = "label",
    id_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    top_n_heatmap: int = 20,
    random_state: int = 42,
    tsne_sample_cap: int = 2500,
    show: bool = True,
) -> dict[str, pd.DataFrame | list[Path] | dict[str, np.ndarray | float]]:
    """Run diagnostics, distributions, correlation, PCA, and t-SNE in one call."""

    if id_cols is None:
        id_cols = DEFAULT_ID_COLS

    if feature_cols is None:
        feature_cols = get_feature_columns(features_df, id_cols=id_cols)
    label_summary = print_feature_diagnostics(features_df, feature_cols=feature_cols, label_col=label_col)
    dist_files = plot_feature_distributions(
        features_df=features_df,
        viz_features=viz_features,
        reports_dir=reports_dir,
        label_col=label_col,
        show=show,
    )
    corr_df = analyze_feature_correlations(
        features_df=features_df,
        feature_cols=feature_cols,
        reports_dir=reports_dir,
        top_n_heatmap=top_n_heatmap,
        show=show,
    )
    dr_stats = plot_dimensionality_reduction(
        features_df=features_df,
        feature_cols=feature_cols,
        label_col=label_col,
        reports_dir=reports_dir,
        random_state=random_state,
        tsne_sample_cap=tsne_sample_cap,
        show=show,
    )
    return {
        "label_summary": label_summary,
        "distribution_files": dist_files,
        "high_correlations": corr_df,
        "dr_stats": dr_stats,
    }
