from __future__ import annotations

"""Preprocessing helpers for IMU + odometry runs.

This module owns:
- QC summaries
- timeline synchronization/resampling
- motion metrics and stationary masking
- leakage-safe split helpers (run-level)
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def resample_sensors(
    acc: pd.DataFrame,
    gyro: pd.DataFrame,
    odo: pd.DataFrame,
    target_hz: float = 100.0,
    tcol: str = "t_rel",
    method: str = "linear",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resample accelerometer, gyroscope, and odometry to a common sample rate.

    Creates a unified time grid spanning all three sensors and interpolates each
    to that grid.  The label column, if present, is forward-filled (nearest prior
    label) rather than interpolated.

    Args:
        acc: Accelerometer DataFrame with a time column and signal columns.
        gyro: Gyroscope DataFrame.
        odo: Odometry DataFrame.
        target_hz: Target sample rate in Hz (default: 100).
        tcol: Name of the time column (default: 't_rel').
        method: Interpolation method — ``'linear'`` or ``'nearest'``.

    Returns:
        Tuple of ``(acc_resampled, gyro_resampled, odo_resampled)``.
    """
    t_min = min(acc[tcol].min(), gyro[tcol].min(), odo[tcol].min())
    t_max = max(acc[tcol].max(), gyro[tcol].max(), odo[tcol].max())
    dt = 1.0 / target_hz
    unified_time = np.arange(t_min, t_max + dt / 2, dt)

    def _interpolate(df: pd.DataFrame) -> pd.DataFrame:
        label_col = "label" if "label" in df.columns else None
        numeric_cols = [c for c in df.columns if c != tcol and c != label_col]

        data: dict = {}
        for col in numeric_cols:
            if method == "linear":
                data[col] = np.interp(unified_time, df[tcol].to_numpy(float), df[col].to_numpy(float))
            elif method == "nearest":
                idx = np.clip(np.searchsorted(df[tcol], unified_time), 0, len(df) - 1)
                data[col] = df[col].iloc[idx].to_numpy()
            else:
                raise ValueError(f"Unknown interpolation method: {method!r}")

        if label_col:
            idx = np.clip(np.searchsorted(df[tcol], unified_time, side="right") - 1, 0, len(df) - 1)
            data[label_col] = df[label_col].iloc[idx].to_numpy()

        result = pd.DataFrame(data)
        result.insert(0, tcol, unified_time)
        return result.reset_index(drop=True)

    return _interpolate(acc), _interpolate(gyro), _interpolate(odo)


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


def plot_sanity(
    acc: pd.DataFrame,
    gyro: pd.DataFrame,
    odo: pd.DataFrame,
    tcol: str = "t_rel",
    show: bool = True,
) -> plt.Figure:
    """Plot per-sensor sanity panels for one run."""

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="col")

    axes[0, 0].plot(acc[tcol], acc["ax"], label="ax")
    axes[0, 0].plot(acc[tcol], acc["ay"], label="ay")
    axes[0, 0].plot(acc[tcol], acc["az"], label="az")
    axes[0, 0].set_title("Accelerometer axes")
    axes[0, 0].set_ylabel("acc [g]")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    acc_mag = np.sqrt(acc["ax"] ** 2 + acc["ay"] ** 2 + acc["az"] ** 2)
    axes[0, 1].plot(acc[tcol], acc_mag)
    axes[0, 1].set_title("Accelerometer magnitude")
    axes[0, 1].set_ylabel("|a| [g]")
    axes[0, 1].grid(True)

    axes[1, 0].plot(gyro[tcol], gyro["gx"], label="gx")
    axes[1, 0].plot(gyro[tcol], gyro["gy"], label="gy")
    axes[1, 0].plot(gyro[tcol], gyro["gz"], label="gz")
    axes[1, 0].set_title("Gyroscope axes")
    axes[1, 0].set_ylabel("gyro [deg/s]")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    gyro_mag = np.sqrt(gyro["gx"] ** 2 + gyro["gy"] ** 2 + gyro["gz"] ** 2)
    axes[1, 1].plot(gyro[tcol], gyro_mag)
    axes[1, 1].set_title("Gyroscope magnitude")
    axes[1, 1].set_ylabel("|g| [deg/s]")
    axes[1, 1].grid(True)

    axes[2, 0].plot(odo[tcol], odo["v1"], label="v1")
    axes[2, 0].plot(odo[tcol], odo["v2"], label="v2")
    axes[2, 0].set_title("Odometry channels")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_ylabel("velocity [m/s]")
    axes[2, 0].grid(True)
    axes[2, 0].legend()

    speed = 0.5 * (odo["v1"] + odo["v2"])
    axes[2, 1].plot(odo[tcol], speed)
    axes[2, 1].set_title("Odometry speed")
    axes[2, 1].set_xlabel("time [s]")
    axes[2, 1].set_ylabel("speed [m/s]")
    axes[2, 1].grid(True)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_dt_diagnostics(
    sensor_name: str,
    df: pd.DataFrame,
    tcol: str = "t_rel",
    bins: int = 80,
    show: bool = True,
) -> plt.Figure | None:
    """Plot dt histogram and dt-vs-time for one sensor frame."""

    if len(df) < 2:
        return None

    dt_series = df[tcol].diff().dropna()
    t_for_dt = df[tcol].iloc[1:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(dt_series, bins=bins)
    axes[0].set_title(f"{sensor_name}: dt histogram")
    axes[0].set_xlabel("dt [s]")
    axes[0].set_ylabel("count")
    axes[0].grid(True)

    axes[1].plot(t_for_dt, dt_series, linewidth=0.8)
    axes[1].set_title(f"{sensor_name}: dt over {tcol}")
    axes[1].set_xlabel(f"{tcol} [s]")
    axes[1].set_ylabel("dt [s]")
    axes[1].grid(True)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def bias_from_first_seconds(
    df: pd.DataFrame,
    cols: list[str],
    tcol: str = "t_rel",
    seconds: float = 5.0,
) -> pd.Series:
    """Estimate channel means over an initial time window."""

    seg = df[df[tcol] <= seconds]
    if seg.empty:
        return pd.Series(index=cols, dtype=float)
    return seg[cols].mean()


def time_range(df: pd.DataFrame, tcol: str = "t_rel") -> tuple[float, float]:
    """Return min/max time for a frame."""

    return float(df[tcol].min()), float(df[tcol].max())


def sensor_channel_checks(
    df: pd.DataFrame,
    channels: list[str],
    sensor_name: str,
) -> pd.DataFrame:
    """Check value ranges and potential clipping/frozen-sample symptoms."""

    rows = []
    for ch in channels:
        s = df[ch].dropna()
        if s.empty:
            rows.append(
                {
                    "sensor": sensor_name,
                    "channel": ch,
                    "max_abs": float("nan"),
                    "n": 0,
                    "n_unique": 0,
                    "repeat_ratio": float("nan"),
                    "longest_constant_run": 0,
                    "clip_candidate_count": 0,
                    "flag_frozen": True,
                    "flag_flat_top": False,
                }
            )
            continue

        arr = s.to_numpy(dtype=float)
        n = int(arr.size)
        n_unique = int(pd.Series(arr).nunique(dropna=True))
        same_as_prev = np.r_[False, np.isclose(np.diff(arr), 0.0)]
        repeat_ratio = float(np.mean(same_as_prev))
        max_abs = float(np.nanmax(np.abs(arr)))

        near_top = np.abs(np.abs(arr) - max_abs) <= max(1e-9, 0.001 * max_abs)
        clip_candidate_count = int(np.sum(near_top))

        longest = 1
        cur = 1
        for i in range(1, n):
            if np.isclose(arr[i], arr[i - 1]):
                cur += 1
                if cur > longest:
                    longest = cur
            else:
                cur = 1

        flag_frozen = bool(n_unique <= 2 or repeat_ratio > 0.98 or longest >= max(20, int(0.1 * n)))
        flag_flat_top = bool(clip_candidate_count >= max(10, int(0.01 * n)))

        rows.append(
            {
                "sensor": sensor_name,
                "channel": ch,
                "max_abs": max_abs,
                "n": n,
                "n_unique": n_unique,
                "repeat_ratio": repeat_ratio,
                "longest_constant_run": int(longest),
                "clip_candidate_count": clip_candidate_count,
                "flag_frozen": flag_frozen,
                "flag_flat_top": flag_flat_top,
            }
        )

    return pd.DataFrame(rows)


def build_dataset_qc_table(
    runs: list,
    tcol: str = "t_rel",
    min_duration_s: float = 30.0,
    expected_fs_ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Build a run-level QC table with flattened sensor metrics and hard pass/fail flags."""

    if expected_fs_ranges is None:
        expected_fs_ranges = {
            "acc": (70.0, 100.0),
            "gyro": (70.0, 100.0),
            "odo": (100.0, 150.0),
        }

    rows = []
    for run in runs:
        per_sensor = qc_report(run.sensors(), tcol=tcol).set_index("sensor")
        wide = per_sensor.stack().rename("value").reset_index()
        wide["col"] = wide["sensor"] + "__" + wide["level_1"]
        row = wide.set_index("col")["value"].to_dict()

        gap_ok = (per_sensor["n_gaps_gt_factor_times_med"].fillna(1) == 0).all()
        dur_ok = (per_sensor["duration_s"].fillna(0) >= min_duration_s).all()

        fs_checks = []
        for sensor, stats in per_sensor.iterrows():
            fs = float(stats["fs_med_hz"])
            lo, hi = expected_fs_ranges.get(sensor, (0.0, float("inf")))
            fs_checks.append(lo <= fs <= hi)
        fs_ok = all(fs_checks) if fs_checks else False

        row["run_id"] = run.run_id
        row["pass_dt_gaps"] = bool(gap_ok)
        row["pass_duration"] = bool(dur_ok)
        row["pass_fs"] = bool(fs_ok)
        row["pass_all"] = bool(gap_ok and dur_ok and fs_ok)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("run_id").sort_index()


def split_qc_pass_fail(
    dataset_qc: pd.DataFrame, pass_col: str = "pass_all"
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Split a run-level QC table into passing/failing subsets and run ID lists."""

    if pass_col not in dataset_qc.columns:
        raise KeyError(f"Missing column: {pass_col}")
    good = dataset_qc[dataset_qc[pass_col]].copy()
    bad = dataset_qc[~dataset_qc[pass_col]].copy()
    return good, bad, good.index.tolist(), bad.index.tolist()


def save_qc_outputs(
    runs: list,
    raw_root: str | Path,
    out_root: str | Path,
    tcol: str = "t_rel",
    dpi: int = 150,
) -> None:
    """Save per-run QC report JSON and sanity plot, mirroring raw-root folder labels."""

    raw_root = Path(raw_root).resolve()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for run in runs:
        rel = run.run_dir.resolve().relative_to(raw_root)
        run_out = out_root / rel
        run_out.mkdir(parents=True, exist_ok=True)

        report_df = qc_report(run.sensors(), tcol=tcol)
        report_dict = report_df.set_index("sensor").to_dict(orient="index")
        with (run_out / "qc_report.json").open("w") as f:
            json.dump(
                {"run_id": run.run_id, "source_run_dir": str(run.run_dir), "qc_report": report_dict},
                f,
                indent=2,
            )

        fig = plot_sanity(run.acc, run.gyro, run.odo, tcol=tcol, show=False)
        fig.savefig(run_out / "sanity.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
