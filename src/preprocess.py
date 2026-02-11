from __future__ import annotations

"""Preprocessing helpers for IMU + odometry runs.

This module owns:
- QC summaries
- timeline synchronization/resampling
- motion metrics and stationary masking
- leakage-safe split helpers (run-level)
"""

from typing import Iterable

import numpy as np
import pandas as pd


def qc_stats(df: pd.DataFrame, tcol: str = "t_rel", gap_factor: float = 5.0) -> dict[str, float | int]:
    """Compute compact timing/QC statistics for a sensor frame.

    These stats are aimed at spotting timestamp issues (gaps, jitter, low rate).
    """

    if len(df) < 2:
        return {
            "n": int(len(df)),
            "duration_s": 0.0,
            "fs_med_hz": float("nan"),
            "dt_med": float("nan"),
            "dt_p99": float("nan"),
            "dt_max": float("nan"),
            "gap_ratio_dtmax_over_dtmed": float("nan"),
            "n_gaps_gt_factor_times_med": 0,
        }

    dt = df[tcol].diff().dropna().to_numpy(dtype=float)
    med = float(np.median(dt))
    dt_max = float(np.max(dt))
    return {
        # `duration_s` assumes the frame is sorted by `tcol` (enforced at load time).
        "n": int(len(df)),
        "duration_s": float(df[tcol].iloc[-1] - df[tcol].iloc[0]),
        "fs_med_hz": float(1.0 / med) if med > 0 else float("nan"),
        "dt_med": med,
        "dt_p99": float(np.quantile(dt, 0.99)),
        "dt_max": dt_max,
        "gap_ratio_dtmax_over_dtmed": float(dt_max / med) if med > 0 else float("nan"),
        "n_gaps_gt_factor_times_med": int(np.sum(dt > gap_factor * med)) if med > 0 else 0,
    }


def qc_report(sensor_frames: dict[str, pd.DataFrame], tcol: str = "t_rel") -> pd.DataFrame:
    """Build a table of `qc_stats` rows for each named sensor frame."""

    rows = []
    for sensor_name, frame in sensor_frames.items():
        row = {"sensor": sensor_name}
        row.update(qc_stats(frame, tcol=tcol))
        rows.append(row)
    return pd.DataFrame(rows)


def infer_sampling_rate(df: pd.DataFrame, tcol: str = "t_rel") -> float:
    """Infer sampling frequency from the median delta time.

    Median is used instead of mean because it is less sensitive to occasional
    dropped packets/gaps.
    """

    if len(df) < 2:
        raise ValueError("Need at least 2 rows to infer sampling rate.")
    dt = df[tcol].diff().dropna().to_numpy(dtype=float)
    med_dt = float(np.median(dt))
    if med_dt <= 0:
        raise ValueError("Invalid time deltas; median dt must be > 0.")
    return 1.0 / med_dt


def _interp_to_grid(
    df: pd.DataFrame,
    value_cols: Iterable[str],
    target_t: np.ndarray,
    tcol: str = "t_rel",
) -> pd.DataFrame:
    """Interpolate selected columns onto a shared target time grid."""

    src_t = df[tcol].to_numpy(dtype=float)
    out = pd.DataFrame({"t_rel": target_t})
    for col in value_cols:
        src_y = df[col].to_numpy(dtype=float)
        out[col] = np.interp(target_t, src_t, src_y, left=np.nan, right=np.nan)
    return out


def sync_and_resample(
    acc: pd.DataFrame,
    gyro: pd.DataFrame,
    odo: pd.DataFrame,
    pose: pd.DataFrame | None = None,
    fs_hz: float = 100.0,
    tcol: str = "t_rel",
) -> pd.DataFrame:
    """Synchronize sensors and resample them to a fixed sample rate.

    Strategy:
    - Use only the overlapping time interval across available sensors.
    - Build a uniform grid at `fs_hz`.
    - Interpolate each numeric channel onto that grid.
    """

    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive.")

    sources = [acc, gyro, odo] + ([pose] if pose is not None else [])
    # Crop to strict overlap to avoid extrapolation artifacts at boundaries.
    t_start = max(float(df[tcol].iloc[0]) for df in sources)
    t_end = min(float(df[tcol].iloc[-1]) for df in sources)
    if t_end <= t_start:
        raise ValueError("No overlapping time range found across sensors.")

    dt = 1.0 / fs_hz
    # +0.5*dt keeps the last expected grid point when floating-point rounding appears.
    target_t = np.arange(t_start, t_end + 0.5 * dt, dt, dtype=float)
    aligned = pd.DataFrame({"t_rel": target_t})

    acc_interp = _interp_to_grid(acc, ["ax", "ay", "az"], target_t, tcol=tcol)
    gyro_interp = _interp_to_grid(gyro, ["gx", "gy", "gz"], target_t, tcol=tcol)
    odo_interp = _interp_to_grid(odo, ["v1", "v2"], target_t, tcol=tcol)

    aligned = aligned.merge(acc_interp, on="t_rel", how="left")
    aligned = aligned.merge(gyro_interp, on="t_rel", how="left")
    aligned = aligned.merge(odo_interp, on="t_rel", how="left")

    if pose is not None:
        # Some logs may omit tilt; include only pose columns that exist.
        pose_cols = [col for col in ["x", "y", "heading", "tilt"] if col in pose.columns]
        pose_interp = _interp_to_grid(pose, pose_cols, target_t, tcol=tcol)
        aligned = aligned.merge(pose_interp, on="t_rel", how="left")

    return aligned


def add_motion_metrics(
    df: pd.DataFrame,
    acc_cols: tuple[str, str, str] = ("ax", "ay", "az"),
    gyro_cols: tuple[str, str, str] = ("gx", "gy", "gz"),
    odo_cols: tuple[str, str] = ("v1", "v2"),
    gravity: float = 1.0,
) -> pd.DataFrame:
    """Add derived motion descriptors used in QC and stationary detection.

    Assumption:
    - Accelerometer channels are in units of g by default (`gravity=1.0`).
    """

    out = df.copy()

    acc_xyz = out.loc[:, list(acc_cols)].to_numpy(dtype=float)
    gyro_xyz = out.loc[:, list(gyro_cols)].to_numpy(dtype=float)
    out["acc_mag"] = np.linalg.norm(acc_xyz, axis=1)
    out["acc_dev"] = np.abs(out["acc_mag"] - gravity)
    out["gyro_mag"] = np.linalg.norm(gyro_xyz, axis=1)

    if odo_cols[0] in out.columns and odo_cols[1] in out.columns:
        # Average wheel velocity as a simple forward-speed proxy.
        out["speed"] = 0.5 * (out[odo_cols[0]] + out[odo_cols[1]])
        out["speed_abs"] = out["speed"].abs()
        # Wheel mismatch can later help identify slip/turning anomalies.
        out["wheel_diff_abs"] = (out[odo_cols[0]] - out[odo_cols[1]]).abs()

    return out


def _enforce_min_true_segment_length(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Remove short True bursts from a boolean mask.

    This acts like temporal debouncing for event segments.
    """

    if min_len <= 1:
        return mask.astype(bool)

    mask = mask.astype(bool)
    n = len(mask)
    out = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        # Find contiguous True segment [i, j).
        j = i + 1
        while j < n and mask[j]:
            j += 1
        # Keep only segments that meet minimum length.
        if (j - i) >= min_len:
            out[i:j] = True
        i = j

    return out


def detect_stationary(
    df: pd.DataFrame,
    fs_hz: float | None = None,
    gyro_mag_max: float = 2.0,
    acc_dev_max: float = 0.05,
    speed_abs_max: float = 0.03,
    min_duration_s: float = 0.5,
    tcol: str = "t_rel",
) -> pd.DataFrame:
    """Flag stationary samples using IMU + odometry thresholds.

    A sample is stationary when all enabled conditions pass:
    - low angular-rate magnitude
    - accelerometer magnitude close to gravity
    - low wheel speed (if odometry present)

    `is_stationary_raw` stores the pointwise threshold result.
    `is_stationary` applies a minimum-duration filter to reduce flicker.
    """

    out = add_motion_metrics(df)
    fs_est = fs_hz if fs_hz is not None else infer_sampling_rate(out, tcol=tcol)
    min_samples = max(1, int(round(min_duration_s * fs_est)))

    conds = [
        out["gyro_mag"] <= gyro_mag_max,
        out["acc_dev"] <= acc_dev_max,
    ]
    # Keep odometry optional so this works for IMU-only logs too.
    if "speed_abs" in out.columns:
        conds.append(out["speed_abs"] <= speed_abs_max)

    base_stationary = np.logical_and.reduce(conds)
    out["is_stationary_raw"] = base_stationary.astype(bool)
    out["is_stationary"] = _enforce_min_true_segment_length(base_stationary, min_samples)
    return out


def remove_stationary(df: pd.DataFrame, stationary_col: str = "is_stationary") -> pd.DataFrame:
    """Return only non-stationary rows."""

    if stationary_col not in df.columns:
        raise KeyError(f"Missing column: {stationary_col}")
    return df.loc[~df[stationary_col]].reset_index(drop=True)


def split_run_ids(
    run_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> dict[str, list[str]]:
    """Split run IDs into train/val/test groups.

    Split is done at run granularity to prevent window-level leakage.
    """

    if len(run_ids) < 3:
        raise ValueError("Need at least 3 runs for train/val/test split.")

    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    rng = np.random.default_rng(random_state)
    shuffled = np.array(run_ids, dtype=object).copy()
    rng.shuffle(shuffled)
    n = len(shuffled)

    n_test = max(1, int(round(test_ratio * n)))
    n_val = max(1, int(round(val_ratio * n)))
    n_train = n - n_test - n_val

    if n_train < 1:
        # Guard small-N edge cases by forcing at least one training run.
        n_train = 1
        overflow = (n_test + n_val + n_train) - n
        if overflow > 0:
            n_test = max(1, n_test - overflow)
        if (n_test + n_val + n_train) > n:
            n_val = max(1, n_val - 1)

    train_ids = shuffled[:n_train].tolist()
    val_ids = shuffled[n_train : n_train + n_val].tolist()
    test_ids = shuffled[n_train + n_val :].tolist()

    if not train_ids or not val_ids or not test_ids:
        raise ValueError("Unable to allocate non-empty train/val/test splits.")

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def assign_split_by_run(
    df: pd.DataFrame,
    split_map: dict[str, list[str]],
    run_col: str = "run_id",
) -> pd.DataFrame:
    """Attach a split label (`train`/`val`/`test`) based on run ID."""

    out = df.copy()
    to_split = {}
    for split_name, runs in split_map.items():
        for run_id in runs:
            to_split[run_id] = split_name
    out["split"] = out[run_col].map(to_split).fillna("unassigned")
    return out
