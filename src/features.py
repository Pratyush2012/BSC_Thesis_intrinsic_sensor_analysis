"""
features.py — Terrain classification feature engineering for wheeled robot IMU data.

Physical axis convention (hardcoded throughout):
    Accelerometer
        ax  →  left / right  (lateral)
        ay  →  up   / down   (vertical, gravity axis — raw ay ≈ −1.0 g at rest)
        az  →  front/ back   (longitudinal)

    Gyroscope
        gx  →  pitch  (nose-up / nose-down rotation around the left-right axis)
        gy  →  yaw    (turning left / right around the up-down axis)
        gz  →  roll   (tilting left / right around the front-back axis)

Gravity removal:
    The per-window DC mean is subtracted from ALL three accelerometer axes.
    This removes the ~−1 g offset on ay and any residual DC bias on ax / az,
    leaving only dynamic (terrain-driven) acceleration in every axis.

Terrain classes in this project:
    dry_dirt_track | grass | muddy_dirt_track | smooth_terrain

Public API
----------
    compute_window_features(windows_df, signals_df, ...) → features_df
    get_feature_columns(features_df)                     → list[str]
    split_velocity_regime_datasets(df, ...)              → (dataset_A, dataset_B)
    apply_log_transform(df)                              → df
    compute_mutual_information(features_df, ...)         → mi_df
    drop_high_correlation_features(features_df, ...)     → (pruned_df, dropped)
    run_feature_evaluation(features_df, ...)             → dict
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── Column name constants ────────────────────────────────────────────────────
ACC_COLS  = ["ax", "ay", "az"]
GYRO_COLS = ["gx", "gy", "gz"]
IMU_COLS  = ACC_COLS + GYRO_COLS

# Columns that are metadata, not model features
DEFAULT_ID_COLS = [
    "window_id", "run_id", "segment_id", "label",
    "t_start", "t_end", "n_samples",
]

# Frequency bands (Hz) used for band-power features
DEFAULT_BANDS = [
    (1.0,  5.0,  "1_5Hz"),    # macro terrain shape / low-speed bumps
    (5.0,  20.0, "5_20Hz"),   # terrain texture — most discriminative band
    (20.0, 40.0, "20_40Hz"),  # vibration from wheel-surface interaction
    (40.0, 80.0, "40_80Hz"),  # high-frequency slip / hard surface ringing
]

# Features with heavy right tails — benefit from log1p before modelling
LOG_TRANSFORM_CANDIDATES = [
    "acc_mag_bandpower_5_20Hz",
    "acc_mag_bandpower_20_40Hz",
    "acc_mag_bandpower_40_80Hz",
    "gyro_mag_bandpower_5_20Hz",
    "gyro_mag_bandpower_20_40Hz",
    "gyro_mag_bandpower_40_80Hz",
    "gyro_mag_energy",
    "acc_mag_energy",
]


# ── Internal helpers ─────────────────────────────────────────────────────────

def _require_seaborn():
    try:
        return __import__("seaborn")
    except ImportError as exc:
        raise ImportError("Plotting functions require seaborn. pip install seaborn.") from exc


def _as_float(x) -> np.ndarray:
    """Convert any array-like to a 1-D float64 numpy array."""
    arr = np.asarray(x, dtype=float)
    return arr.ravel() if arr.ndim != 1 else arr


def _check_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {df_name}: {missing}")


# ── Low-level signal feature functions ──────────────────────────────────────

def zero_crossing_rate(x) -> float:
    """Fraction of adjacent-sample sign changes (normalised to [0, 1])."""
    arr = _as_float(x)
    if len(arr) < 2:
        return np.nan
    signs = np.sign(arr)
    # Replace zeros with the previous sign so they don't inflate the count
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    crossings = int(np.sum(signs[1:] * signs[:-1] < 0))
    return crossings / (len(arr) - 1)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; returns 0.0 when either signal is constant."""
    if len(x) < 2 or len(y) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def band_power(x, fs: int, f_low: float, f_high: float) -> float:
    """
    FFT-based power in the frequency band [f_low, f_high] Hz.
    Signal is DC-removed before the FFT.
    """
    arr = _as_float(x)
    if len(arr) < 2:
        return np.nan
    arr_dc = arr - arr.mean()
    n      = len(arr_dc)
    freqs  = np.fft.rfftfreq(n, d=1.0 / fs)
    psd    = (np.abs(np.fft.rfft(arr_dc)) ** 2) / n
    mask   = (freqs >= f_low) & (freqs <= f_high)
    return float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0


def spectral_features(x, fs: int) -> dict[str, float]:
    """
    FFT spectral summary:
        spectral_energy    — total power across all positive frequencies
        dominant_freq      — frequency with highest PSD (Hz)
        spectral_centroid  — power-weighted mean frequency (Hz)
        spectral_entropy   — Shannon entropy of the normalised PSD
    """
    arr = _as_float(x)
    nan_result = dict(
        spectral_energy=np.nan, dominant_freq=np.nan,
        spectral_centroid=np.nan, spectral_entropy=np.nan,
    )
    if len(arr) < 2:
        return nan_result

    arr_dc = arr - arr.mean()
    n      = len(arr_dc)
    freqs  = np.fft.rfftfreq(n, d=1.0 / fs)
    psd    = (np.abs(np.fft.rfft(arr_dc)) ** 2) / n
    total  = float(np.trapezoid(psd, freqs))

    pos = freqs > 0
    fp, pp = freqs[pos], psd[pos]
    if len(pp) == 0 or pp.sum() <= 0:
        return dict(spectral_energy=total, dominant_freq=np.nan,
                    spectral_centroid=np.nan, spectral_entropy=np.nan)

    dominant  = float(fp[np.argmax(pp)])
    centroid  = float(np.sum(fp * pp) / np.sum(pp))
    p_norm    = pp / pp.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p_norm > 0, np.log2(p_norm), 0.0)
    entropy = float(-np.sum(p_norm * log_p))

    return dict(spectral_energy=total, dominant_freq=dominant,
                spectral_centroid=centroid, spectral_entropy=entropy)


def time_domain_features(x) -> dict[str, float]:
    """
    Statistical and energy time-domain features:
        mean, std, min, max, range, median, iqr,
        skewness, kurtosis, zero_crossing_rate,
        rms, energy, mean_abs
    """
    arr = _as_float(x)
    if len(arr) == 0:
        return {k: np.nan for k in [
            "mean", "std", "min", "max", "range", "median", "iqr",
            "skewness", "kurtosis", "zero_crossing_rate",
            "rms", "energy", "mean_abs",
        ]}
    lo, hi = float(arr.min()), float(arr.max())
    return {
        "mean":               float(arr.mean()),
        "std":                float(arr.std()),
        "min":                lo,
        "max":                hi,
        "range":              hi - lo,
        "median":             float(np.median(arr)),
        "iqr":                float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
        "skewness":           float(sp_stats.skew(arr, bias=False))     if len(arr) >= 3 else np.nan,
        "kurtosis":           float(sp_stats.kurtosis(arr, bias=False)) if len(arr) >= 4 else np.nan,
        "zero_crossing_rate": zero_crossing_rate(arr),
        "rms":                float(np.sqrt(np.mean(arr ** 2))),
        "energy":             float(np.sum(arr ** 2)),
        "mean_abs":           float(np.mean(np.abs(arr))),
    }


def jerk_rms(x, fs: int) -> float:
    """
    RMS of the jerk signal (first difference × fs).
    Jerk captures abrupt changes in acceleration — high on rough terrain.
    """
    arr = _as_float(x)
    if len(arr) < 2:
        return np.nan
    j = np.diff(arr) * fs
    return float(np.sqrt(np.mean(j ** 2)))


def hjorth_params(x) -> dict[str, float]:
    """
    Hjorth activity, mobility, and complexity.
        activity   — signal variance (power proxy)
        mobility   — std(first-diff) / std(signal)  (mean frequency proxy)
        complexity — mobility(first-diff) / mobility(signal)  (bandwidth proxy)
    """
    arr = _as_float(x)
    nan_out = dict(hjorth_activity=np.nan, hjorth_mobility=np.nan, hjorth_complexity=np.nan)
    if len(arr) < 3:
        return nan_out

    var_x   = float(np.var(arr))
    std_x   = float(np.std(arr))
    d1      = np.diff(arr)
    std_d1  = float(np.std(d1))
    activity = var_x

    if std_x < 1e-12:
        return dict(hjorth_activity=activity, hjorth_mobility=0.0, hjorth_complexity=0.0)

    mobility = std_d1 / std_x

    if len(d1) < 2 or std_d1 < 1e-12:
        complexity = 0.0
    else:
        std_d2    = float(np.std(np.diff(d1)))
        mob_d1    = std_d2 / std_d1
        complexity = mob_d1 / mobility if mobility > 1e-12 else 0.0

    return dict(
        hjorth_activity=activity,
        hjorth_mobility=float(mobility),
        hjorth_complexity=float(complexity),
    )


def _per_axis_features(sig: np.ndarray, fs: int, bands: list) -> dict[str, float]:
    """
    Full per-axis feature vector:
        time-domain + FFT spectral + band-power + Hjorth + jerk_rms
    """
    out = {**time_domain_features(sig), **spectral_features(sig, fs)}
    for f_low, f_high, suffix in bands:
        out[f"bandpower_{suffix}"] = band_power(sig, fs, f_low, f_high)
    out.update(hjorth_params(sig))
    out["jerk_rms"] = jerk_rms(sig, fs)
    return out


# ── Main feature extraction ──────────────────────────────────────────────────

def compute_window_features(
    windows_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    sample_rate: int = 100,
    bands: list | None = None,
    id_cols: list[str] | None = None,
    vel_col: str | None = "net_speed",
) -> pd.DataFrame:
    """
    Compute the full window-level feature table from raw IMU signals.

    Parameters
    ----------
    windows_df  : window metadata — must contain:
                      window_id, run_id, segment_id, label, t_start, t_end
    signals_df  : raw signal rows — must contain:
                      run_id, t_rel, ax, ay, az, gx, gy, gz
                  and optionally the velocity column (vel_col).
    sample_rate : sensor sampling rate in Hz (default 100).
    bands       : list of (f_low, f_high, label) tuples for band-power.
                  Defaults to DEFAULT_BANDS.
    id_cols     : metadata columns excluded from feature list.
    vel_col     : velocity column in signals_df for odometry features.
                  Set to None to skip.

    Feature groups computed
    -----------------------
    Per IMU axis (ax, ay, az, gx, gy, gz) and magnitude (acc_mag, gyro_mag):
        - Time domain:  mean, std, min, max, range, median, iqr, skewness,
                        kurtosis, zero_crossing_rate, rms, energy, mean_abs
        - Spectral:     spectral_energy, dominant_freq, spectral_centroid,
                        spectral_entropy
        - Band-power:   bandpower_1_5Hz, bandpower_5_20Hz, bandpower_20_40Hz,
                        bandpower_40_80Hz
        - Hjorth:       hjorth_activity, hjorth_mobility, hjorth_complexity
        - Jerk:         jerk_rms

    Within-sensor correlations:
        acc:  corr_xy (lateral-vertical), corr_xz (lateral-longitudinal),
              corr_yz (vertical-longitudinal)
        gyro: corr_xy (pitch-yaw), corr_xz (pitch-roll), corr_yz (yaw-roll)

    Cross-sensor correlations (IMU coupling — terrain discriminative):
        ay vs gx — vertical bounce × pitch  (robot nose dipping into soft terrain)
        ay vs gz — vertical bounce × roll   (uneven left-right wheel sinkage)
        az vs gx — longitudinal acc × pitch (going over bumps)
        ax vs gz — lateral acc × roll       (side-to-side tipping)

    Odometry (when vel_col is present):
        odom_mean_speed, odom_speed_std, odom_distance

    Gravity / DC removal
    --------------------
    The per-window mean is subtracted from ax, ay, and az before any feature
    computation.  This removes the ≈ −1 g DC offset on ay (vertical / gravity
    axis) and any residual bias on the other axes, leaving only dynamic
    (terrain-driven) acceleration.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    if id_cols is None:
        id_cols = DEFAULT_ID_COLS

    _check_columns(signals_df, ["run_id", "t_rel"] + IMU_COLS, "signals_df")
    _check_columns(
        windows_df,
        ["window_id", "run_id", "segment_id", "label", "t_start", "t_end"],
        "windows_df",
    )

    has_odom = vel_col is not None and vel_col in signals_df.columns
    if vel_col is not None and not has_odom:
        print(f"[WARNING] vel_col='{vel_col}' not found in signals_df — odometry features skipped.")

    rows = []

    for _, w in windows_df.iterrows():
        run_id          = w["run_id"]
        t_start, t_end  = float(w["t_start"]), float(w["t_end"])

        # Slice signal rows belonging to this window
        seg = signals_df[
            (signals_df["run_id"] == run_id)
            & (signals_df["t_rel"] >= t_start)
            & (signals_df["t_rel"] <= t_end)
        ]
        if seg.empty:
            continue

        # ── Gravity / DC removal ─────────────────────────────────────────
        # Subtract per-window mean from every accelerometer axis.
        # ay  → removes ≈ −1 g gravity component
        # ax, az → removes any residual DC bias
        acc_raw      = seg[ACC_COLS].to_numpy(dtype=float)
        acc_centered = acc_raw - acc_raw.mean(axis=0, keepdims=True)
        ax_s, ay_s, az_s = acc_centered[:, 0], acc_centered[:, 1], acc_centered[:, 2]

        # Raw gyro signals (no DC removal needed — gyro reads angular rate, not offset)
        gx_s = seg["gx"].to_numpy(dtype=float)
        gy_s = seg["gy"].to_numpy(dtype=float)
        gz_s = seg["gz"].to_numpy(dtype=float)

        row: dict = {
            "window_id":  int(w["window_id"]),
            "run_id":     run_id,
            "segment_id": int(w["segment_id"]),
            "label":      w["label"],
            "t_start":    t_start,
            "t_end":      t_end,
            "n_samples":  int(len(seg)),
        }

        # ── Per-axis accelerometer features ─────────────────────────────
        for col, sig in zip(ACC_COLS, [ax_s, ay_s, az_s]):
            for k, v in _per_axis_features(sig, sample_rate, bands).items():
                row[f"{col}_{k}"] = v

        # ── Per-axis gyroscope features ──────────────────────────────────
        for col, sig in zip(GYRO_COLS, [gx_s, gy_s, gz_s]):
            for k, v in _per_axis_features(sig, sample_rate, bands).items():
                row[f"{col}_{k}"] = v

        # ── Magnitude vector features ─────────────────────────────────
        # ||acc||  and  ||gyro||  capture overall vibration intensity
        acc_mag  = np.sqrt(ax_s**2 + ay_s**2 + az_s**2)
        gyro_mag = np.sqrt(gx_s**2 + gy_s**2 + gz_s**2)

        for prefix, sig in [("acc_mag", acc_mag), ("gyro_mag", gyro_mag)]:
            for k, v in _per_axis_features(sig, sample_rate, bands).items():
                row[f"{prefix}_{k}"] = v

        # ── Within-sensor correlations ────────────────────────────────
        # Accelerometer
        row["acc_corr_xy"] = safe_pearson(ax_s, ay_s)  # lateral  × vertical
        row["acc_corr_xz"] = safe_pearson(ax_s, az_s)  # lateral  × longitudinal
        row["acc_corr_yz"] = safe_pearson(ay_s, az_s)  # vertical × longitudinal

        # Gyroscope
        row["gyro_corr_xy"] = safe_pearson(gx_s, gy_s)  # pitch × yaw
        row["gyro_corr_xz"] = safe_pearson(gx_s, gz_s)  # pitch × roll
        row["gyro_corr_yz"] = safe_pearson(gy_s, gz_s)  # yaw   × roll

        # ── Cross-sensor correlations ─────────────────────────────────
        # These capture coupling between linear and rotational motion —
        # patterns that change with terrain type.
        #
        #  ay × gx : vertical bounce × pitch
        #            High on soft/muddy terrain where the nose dips in.
        #
        #  ay × gz : vertical bounce × roll
        #            High when left/right wheels sink unevenly (mud, ruts).
        #
        #  az × gx : longitudinal acc × pitch
        #            High when robot crests bumps (dirt track features).
        #
        #  ax × gz : lateral acc × roll
        #            High on cambered or side-sloped surfaces.
        row["cross_ay_gx"] = safe_pearson(ay_s, gx_s)
        row["cross_ay_gz"] = safe_pearson(ay_s, gz_s)
        row["cross_az_gx"] = safe_pearson(az_s, gx_s)
        row["cross_ax_gz"] = safe_pearson(ax_s, gz_s)

        # ── Odometry features ─────────────────────────────────────────
        if has_odom:
            vel = seg[vel_col].to_numpy(dtype=float)
            row["odom_mean_speed"] = float(vel.mean())
            row["odom_speed_std"]  = float(vel.std())
            row["odom_distance"]   = float(np.abs(vel).sum() / sample_rate)

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No features were generated. "
            "Check that window t_start/t_end ranges overlap with signals_df t_rel values."
        )

    features_df = (
        pd.DataFrame(rows)
        .sort_values("window_id")
        .reset_index(drop=True)
    )

    # Reorder: id_cols first, then feature columns
    feat_cols = [c for c in features_df.columns if c not in id_cols]
    return features_df[id_cols + feat_cols]


# ── Feature column helpers ───────────────────────────────────────────────────

def get_feature_columns(
    features_df: pd.DataFrame,
    id_cols: list[str] | None = None,
) -> list[str]:
    """Return all columns that are model features (i.e. not in id_cols)."""
    if id_cols is None:
        id_cols = DEFAULT_ID_COLS
    return [c for c in features_df.columns if c not in id_cols]


def get_eda_feature_columns(
    features_df: pd.DataFrame,
    id_cols: list[str] | None = None,
) -> list[str]:
    """
    Return a reduced feature subset suitable for EDA visualisations.
    Drops the noisiest / most redundant features (min, max, range, median,
    iqr, skewness, ZCR, 40-80 Hz bandpower) to keep plots readable.
    """
    all_feats = get_feature_columns(features_df, id_cols)
    drop_tokens = [
        "_min", "_max", "_range", "_median", "_iqr",
        "_skewness", "_zero_crossing_rate",
        "_spectral_energy", "_dominant_freq",
        "_spectral_centroid", "_spectral_entropy",
        "_bandpower_40_80Hz",
    ]
    return [c for c in all_feats if not any(t in c for t in drop_tokens)]


# ── Velocity-regime splitting ────────────────────────────────────────────────

def split_velocity_regime_datasets(
    df: pd.DataFrame,
    vel_col: str = "odom_mean_speed",
    mean_thresh: float = 0.6,
    std_thresh: float = 0.10,
    label_col: str = "label",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split feature table into:
        Dataset A — all windows (full diversity of speeds and conditions)
        Dataset B — low, near-constant-speed windows only

    Dataset B is a speed-controlled subset that isolates terrain signal from
    velocity-induced vibration, making terrain features more comparable across
    terrain types.

    Parameters
    ----------
    vel_col      : per-window mean speed column (produced by compute_window_features).
    mean_thresh  : max mean speed for Dataset B (m/s).
    std_thresh   : max speed std for Dataset B (enforces near-constant speed).
    """
    _check_columns(df, [vel_col, "odom_speed_std", label_col], "features_df")

    dataset_A = df.copy()
    mask      = (df[vel_col] < mean_thresh) & (df["odom_speed_std"] < std_thresh)
    dataset_B = df[mask].copy().reset_index(drop=True)

    if verbose:
        n_total = len(df)
        n_kept  = len(dataset_B)
        sep     = "=" * 62

        print(sep)
        print("DATASET A — All windows")
        print(sep)
        print(f"  Total windows : {n_total}")
        _print_label_dist(dataset_A, label_col, n_total)

        print()
        print(sep)
        print(f"DATASET B — Speed-controlled subset")
        print(f"  Criteria : {vel_col} < {mean_thresh} m/s  AND  odom_speed_std < {std_thresh} m/s")
        print(sep)
        print(f"  Windows kept    : {n_kept:>5}  ({n_kept / n_total * 100:.1f}%)")
        print(f"  Windows removed : {n_total - n_kept:>5}  ({(n_total - n_kept) / n_total * 100:.1f}%)")
        _print_label_dist(dataset_B, label_col, n_kept)

        for cls, cnt in dataset_B[label_col].value_counts().items():
            if cnt < 50:
                print(f"\n  ⚠ CRITICAL: '{cls}' has only {cnt} windows in Dataset B.")
                print("    Consider relaxing velocity thresholds.")
            elif cnt < 100:
                print(f"\n  ⚠ WARNING:  '{cls}' has only {cnt} windows in Dataset B.")

    return dataset_A, dataset_B


def _print_label_dist(df: pd.DataFrame, label_col: str, total: int) -> None:
    print("  Class distribution:")
    for cls, cnt in df[label_col].value_counts().sort_index().items():
        bar = "█" * int(cnt / max(total, 1) * 30)
        print(f"    {cls:<25} {cnt:>5}  ({cnt / total * 100:5.1f}%)  {bar}")


# ── Transforms and feature selection ────────────────────────────────────────

def apply_log_transform(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Apply log1p to heavy-tailed energy / band-power features.
    Defaults to LOG_TRANSFORM_CANDIDATES that are present in df.
    Always clips to ≥ 0 before transforming (negative values → 0 before log1p).
    """
    out = df if inplace else df.copy()
    if cols is None:
        cols = [c for c in LOG_TRANSFORM_CANDIDATES if c in out.columns]
    applied = []
    for c in cols:
        if c in out.columns:
            out[c] = np.log1p(out[c].clip(lower=0.0))
            applied.append(c)
    if applied:
        print(f"[log1p transform] Applied to {len(applied)} columns.")
    return out


def compute_mutual_information(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    label_col: str = "label",
    n_neighbors: int = 5,
    random_state: int = 42,
    top_n: int = 30,
    plot: bool = True,
    reports_dir: str | Path | None = None,
    show: bool = True,
) -> pd.DataFrame:
    """
    Compute mutual information (MI) between each feature and the terrain label.

    Returns a DataFrame sorted by MI score descending.
    Prints and optionally plots the top_n features.

    Note: run this on the training split only in your final ML pipeline
    to avoid data leakage.  For exploratory use on the full dataset,
    the leakage risk is low but worth keeping in mind.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(features_df)

    X = np.nan_to_num(
        features_df[feature_cols].to_numpy(dtype=float),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    y = features_df[label_col].to_numpy()

    scores = mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=random_state)
    mi_df  = (
        pd.DataFrame({"feature": feature_cols, "mi_score": scores})
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )

    n_show = min(top_n, len(mi_df))
    print(f"\nTop {n_show} features by mutual information (vs '{label_col}'):")
    print(mi_df.head(n_show).to_string(index=False))

    if plot:
        top = mi_df.head(n_show)
        fig, ax = plt.subplots(figsize=(9, max(4, n_show * 0.33)))
        ax.barh(top["feature"][::-1], top["mi_score"][::-1], color="steelblue")
        ax.set_xlabel("Mutual Information Score")
        ax.set_title(f"Top {n_show} features — MI vs terrain label")
        fig.tight_layout()
        _maybe_save(fig, reports_dir, "mutual_information.png", show)

    return mi_df


def drop_high_correlation_features(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    threshold: float = 0.95,
    id_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove one feature from each pair whose absolute Pearson correlation
    exceeds `threshold`.  The feature with lower variance is dropped.

    Returns
    -------
    pruned_df : DataFrame with redundant columns removed.
    dropped   : sorted list of dropped column names.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(features_df, id_cols)

    corr     = features_df[feature_cols].corr().abs()
    upper    = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    variances = features_df[feature_cols].var()
    to_drop: set[str] = set()

    for col in upper.columns:
        if col in to_drop:
            continue
        partners = upper.index[upper[col] > threshold].tolist()
        for partner in partners:
            if partner in to_drop:
                continue
            # Keep the one with higher variance (more informative)
            drop = col if variances[col] < variances[partner] else partner
            to_drop.add(drop)

    dropped   = sorted(to_drop)
    pruned_df = features_df.drop(columns=dropped)

    print(f"\n[Correlation pruning] threshold={threshold}")
    print(f"  Dropped {len(dropped)} features:")
    for c in dropped:
        print(f"    {c}")
    print(f"  Features remaining: {pruned_df.shape[1] - len(id_cols or DEFAULT_ID_COLS)}")

    return pruned_df, dropped


# ── Diagnostics ──────────────────────────────────────────────────────────────

def print_feature_diagnostics(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Print a summary of the feature table and return a label-distribution DataFrame.
    Highlights missing values and checks key separator features.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(features_df)

    sep = "=" * 70
    print(sep)
    print("FEATURE TABLE DIAGNOSTICS")
    print(sep)
    print(f"  Shape          : {features_df.shape}")
    print(f"  Feature count  : {len(feature_cols)}")

    nan_counts  = features_df[feature_cols].isna().sum()
    nan_total   = int(nan_counts.sum())
    print(f"  Missing values : {nan_total} total")
    if nan_total > 0:
        print("  Top columns with NaNs:")
        print(
            nan_counts[nan_counts > 0]
            .sort_values(ascending=False)
            .head(10)
            .to_string()
        )

    counts  = features_df[label_col].value_counts().sort_index()
    pcts    = (counts / len(features_df) * 100).round(2)
    summary = pd.DataFrame({"count": counts, "pct_%": pcts})
    print(f"\n  Label distribution ({label_col}):")
    print(summary.to_string())

    # Quick mean-by-label table for key separator features
    key_cols = [c for c in [
        "acc_mag_rms", "acc_mag_std", "acc_mag_energy",
        "gyro_mag_rms", "gyro_mag_std",
        "ay_jerk_rms", "gz_jerk_rms",
        "cross_ay_gx", "cross_ay_gz",
    ] if c in features_df.columns]

    if key_cols:
        print(f"\n  Per-class mean of key features:")
        print(features_df.groupby(label_col)[key_cols].mean().round(4).to_string())

    print(sep)
    return summary


# ── EDA visualisations ───────────────────────────────────────────────────────
# Helper to syncronize the color of the histogram and violion plots
def _build_class_color_map(sns, classes, palette) -> dict:
    colors = sns.color_palette(palette, n_colors=len(classes))
    return dict(zip(classes, colors))


def plot_feature_distributions(
    features_df: pd.DataFrame,
    viz_features: list[str],
    reports_dir: str | Path,
    label_col: str = "label",
    palette: str = "Set2",
    show: bool = True,
) -> list[Path]:
    """
    For each feature in viz_features: save a figure with
        - left panel:  overlapping histograms per terrain class
        - right panel: violin plot per terrain class
    """
    sns = _require_seaborn()
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    present = [c for c in viz_features if c in features_df.columns]
    missing = [c for c in viz_features if c not in features_df.columns]
    if missing:
        print(f"[plot_feature_distributions] Skipping (not in df): {missing}")
    if not present:
        raise RuntimeError("None of the requested viz_features are in features_df.")

    classes   = sorted(features_df[label_col].unique())
    class_to_color = _build_class_color_map(sns, classes, palette)
    out_files = []

    for feat in present:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(feat, fontsize=12, fontweight="bold")

        # Histogram
        axes[0].hist(
            [features_df.loc[features_df[label_col] == cls, feat].dropna() for cls in classes],
            bins=25, alpha=0.55, edgecolor="black",
            label=[str(c) for c in classes],
            color=[class_to_color[cls] for cls in classes]
        )
        axes[0].set_xlabel(feat)
        axes[0].set_ylabel("Count")
        axes[0].set_title("Histogram by terrain class")
        axes[0].legend(fontsize=8)

        # Violin
        sns.violinplot(
            data=features_df, x=label_col, y=feat,
            ax=axes[1], inner="quartile", cut=0, order=classes, palette=class_to_color,
        )
        axes[1].set_title("Violin by terrain class")
        axes[1].tick_params(axis="x", rotation=20)

        fig.tight_layout()
        out_path = out_dir / f"dist_{feat}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        out_files.append(out_path)
        plt.show() if show else plt.close(fig)

    print(f"[plot_feature_distributions] Saved {len(out_files)} figures → {out_dir}")
    return out_files


def plot_correlation_heatmap(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    reports_dir: str | Path | None = None,
    top_n: int = 25,
    show: bool = True,
) -> pd.DataFrame:
    """
    Plot a correlation heatmap of the top_n most variable features.
    Returns a DataFrame of the top-15 absolute correlation pairs.
    """
    sns = _require_seaborn()
    if feature_cols is None:
        feature_cols = get_feature_columns(features_df)

    # Select top_n most variable features for the heatmap
    ranked      = features_df[feature_cols].std(numeric_only=True).sort_values(ascending=False)
    heatmap_cols = ranked.head(min(top_n, len(ranked))).index.tolist()
    corr         = features_df[heatmap_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, cmap="coolwarm", center=0, ax=ax,
        square=True, linewidths=0.3, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Pearson correlation — top {len(heatmap_cols)} variable features")
    fig.tight_layout()
    _maybe_save(fig, reports_dir, "corr_heatmap.png", show)

    # Build ranked pairs list
    cols  = corr.columns.tolist()
    pairs = [
        (cols[i], cols[j], corr.iloc[i, j], abs(corr.iloc[i, j]))
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
    ]
    pairs_df = (
        pd.DataFrame(pairs, columns=["feature_1", "feature_2", "corr", "abs_corr"])
        .sort_values("abs_corr", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    print("\nTop 15 absolute correlation pairs:")
    print(pairs_df[["feature_1", "feature_2", "corr"]].to_string(index=False))
    return pairs_df


def plot_pca_tsne(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    label_col: str = "label",
    reports_dir: str | Path | None = None,
    random_state: int = 42,
    tsne_sample_cap: int = 2500,
    show: bool = True,
) -> dict:
    """
    PCA (2D) and t-SNE (2D) scatter plots coloured by terrain label.
    t-SNE is computed on a stratified subsample (≤ tsne_sample_cap rows)
    for speed.

    Returns a dict with PCA explained variance stats and t-SNE sample count.
    """
    sns = _require_seaborn()
    if feature_cols is None:
        feature_cols = get_feature_columns(features_df)

    X = np.nan_to_num(
        features_df[feature_cols].to_numpy(dtype=float),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    X_scaled = StandardScaler().fit_transform(X)

    # ── PCA ─────────────────────────────────────────────────────────────────
    pca   = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
        label_col: features_df[label_col].values,
    })
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=label_col,
                    alpha=0.7, s=35, ax=ax)
    evr = pca.explained_variance_ratio_
    ax.set_title(
        f"PCA — 2D projection  "
        f"(PC1={evr[0]:.1%}, PC2={evr[1]:.1%}, total={evr.sum():.1%})"
    )
    ax.legend(title="Terrain", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _maybe_save(fig, reports_dir, "pca_2d.png", show)
    print(f"PCA explained variance: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, total={evr.sum():.3f}")

    # ── t-SNE ────────────────────────────────────────────────────────────────
    sample_df = _stratified_sample(features_df, label_col, tsne_sample_cap, random_state)
    tsne_X    = np.nan_to_num(
        sample_df[feature_cols].to_numpy(dtype=float),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    tsne_X      = StandardScaler().fit_transform(tsne_X)
    perplexity  = min(30, max(5, len(tsne_X) - 1))
    X_tsne      = TSNE(
        n_components=2, random_state=random_state,
        perplexity=perplexity, init="pca", learning_rate="auto",
    ).fit_transform(tsne_X)

    tsne_df = pd.DataFrame({
        "TSNE1": X_tsne[:, 0], "TSNE2": X_tsne[:, 1],
        label_col: sample_df[label_col].values,
    })
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue=label_col,
                    alpha=0.75, s=35, ax=ax)
    ax.set_title(f"t-SNE — 2D projection  (n={len(tsne_X)}, perplexity={perplexity})")
    ax.legend(title="Terrain", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _maybe_save(fig, reports_dir, "tsne_2d.png", show)

    return {
        "pca_explained_variance_ratio": evr,
        "pca_cumulative_variance":      float(evr.sum()),
        "n_tsne_samples":               int(len(tsne_X)),
    }


def _stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    cap: int,
    random_state: int,
) -> pd.DataFrame:
    """Stratified subsample keeping class proportions."""
    if len(df) <= cap:
        return df
    sampled = (
        df.groupby(label_col, group_keys=False)
        .apply(
            lambda g: g.sample(
                n=max(1, int(cap * len(g) / len(df))),
                random_state=random_state,
            ),
            include_groups=False,
        )
    )
    if label_col not in sampled.columns:
        sampled = sampled.join(df[[label_col]])
    return sampled.reset_index(drop=True)


def _maybe_save(fig, reports_dir, filename: str, show: bool) -> None:
    if reports_dir is not None:
        p = Path(reports_dir)
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / filename, bbox_inches="tight", dpi=120)
    plt.show() if show else plt.close(fig)


# ── Full EDA pipeline ────────────────────────────────────────────────────────

def run_feature_evaluation(
    features_df: pd.DataFrame,
    reports_dir: str | Path,
    viz_features: list[str],
    label_col: str = "label",
    id_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    top_n_heatmap: int = 25,
    random_state: int = 42,
    tsne_sample_cap: int = 2500,
    show: bool = True,
) -> dict:
    """
    Run the full EDA pipeline:
        1. Feature diagnostics (shape, NaNs, label distribution)
        2. Distribution plots (histogram + violin per feature)
        3. Correlation heatmap + top pairs
        4. PCA and t-SNE projections

    Returns a dict with all results for downstream inspection.
    """
    if id_cols is None:
        id_cols = DEFAULT_ID_COLS
    if feature_cols is None:
        feature_cols = get_feature_columns(features_df, id_cols)

    label_summary = print_feature_diagnostics(features_df, feature_cols, label_col)

    dist_files = plot_feature_distributions(
        features_df, viz_features, reports_dir, label_col, show=show,
    )
    corr_pairs = plot_correlation_heatmap(
        features_df, feature_cols, reports_dir, top_n_heatmap, show,
    )
    dr_stats = plot_pca_tsne(
        features_df, feature_cols, label_col, reports_dir,
        random_state, tsne_sample_cap, show,
    )

    return dict(
        label_summary    = label_summary,
        distribution_files = dist_files,
        correlation_pairs  = corr_pairs,
        dr_stats           = dr_stats,
    )