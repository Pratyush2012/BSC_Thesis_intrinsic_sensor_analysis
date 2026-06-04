from pathlib import Path

import numpy as np
import pandas as pd


def detect_continuous_segments(df, run_col="run_id", tcol="t_rel", gap_threshold=0.025):
    """Return continuous (gap-safe) index segments per run."""
    if run_col not in df.columns or tcol not in df.columns:
        missing = [c for c in [run_col, tcol] if c not in df.columns]
        raise KeyError(f"Missing required columns for segment detection: {missing}")

    segments = []
    for run_id in sorted(df[run_col].dropna().unique()):
        run_df = df[df[run_col] == run_id].sort_values(tcol)
        if run_df.empty:
            continue

        times = run_df[tcol].to_numpy(dtype=float)
        idx = run_df.index.to_numpy()
        dt = np.diff(times)
        gap_positions = np.where(dt > gap_threshold)[0]

        start = 0
        seg_id = 0
        for gpos in gap_positions:
            stop = gpos + 1
            seg_idx = idx[start:stop]
            if len(seg_idx) > 0:
                segments.append(
                    {
                        "run_id": run_id,
                        "segment_id": seg_id,
                        "indices": seg_idx,
                        "start_time": float(times[start]),
                        "end_time": float(times[stop - 1]),
                        "n_samples": int(len(seg_idx)),
                    }
                )
                seg_id += 1
            start = stop

        seg_idx = idx[start:]
        if len(seg_idx) > 0:
            segments.append(
                {
                    "run_id": run_id,
                    "segment_id": seg_id,
                    "indices": seg_idx,
                    "start_time": float(times[start]),
                    "end_time": float(times[-1]),
                    "n_samples": int(len(seg_idx)),
                }
            )

    return segments


def detect_stationary_segments(
    df,
    gyro_cols=("gx", "gy", "gz"),
    acc_cols=("ax", "ay", "az"),
    run_col="run_id",
    tcol="t_rel",
    gap_threshold=0.025,
    gyro_threshold=0.1,
    acc_threshold=0.1,
):
    """Flag continuous segments that are mostly stationary."""
    required = [run_col, tcol] + list(gyro_cols) + list(acc_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for stationary detection: {missing}")

    segments = detect_continuous_segments(df, run_col=run_col, tcol=tcol, gap_threshold=gap_threshold)
    stationary_segments = []

    for seg in segments:
        seg_df = df.loc[seg["indices"]]
        gyro_mag = np.sqrt((seg_df[list(gyro_cols)] ** 2).sum(axis=1))
        acc_centered = seg_df[list(acc_cols)] - seg_df[list(acc_cols)].mean(axis=0)
        acc_dyn = np.sqrt((acc_centered**2).sum(axis=1))

        is_stationary = (gyro_mag.mean() < gyro_threshold) and (acc_dyn.mean() < acc_threshold)
        seg_out = dict(seg)
        seg_out["gyro_mag_mean"] = float(gyro_mag.mean())
        seg_out["acc_dyn_mean"] = float(acc_dyn.mean())
        seg_out["is_stationary"] = bool(is_stationary)
        stationary_segments.append(seg_out)

    return stationary_segments


def remove_stationary(
    df,
    gyro_cols=("gx", "gy", "gz"),
    acc_cols=("ax", "ay", "az"),
    run_col="run_id",
    tcol="t_rel",
    gap_threshold=0.025,
    gyro_threshold=0.1,
    acc_threshold=0.1,
):
    """Remove stationary segments and return (filtered_df, stationarity_table)."""
    seg_table = detect_stationary_segments(
        df,
        gyro_cols=gyro_cols,
        acc_cols=acc_cols,
        run_col=run_col,
        tcol=tcol,
        gap_threshold=gap_threshold,
        gyro_threshold=gyro_threshold,
        acc_threshold=acc_threshold,
    )

    drop_indices = []
    for seg in seg_table:
        if seg["is_stationary"]:
            drop_indices.extend(seg["indices"])

    filtered_df = df.drop(index=drop_indices).copy().sort_values([run_col, tcol]).reset_index(drop=True)
    return filtered_df, pd.DataFrame(seg_table)


def assign_window_labels(labels, strategy="majority"):
    """Return window label from a sequence of labels."""
    s = pd.Series(labels, dtype="string").dropna()
    if s.empty:
        return pd.NA

    if strategy == "last":
        return s.iloc[-1]

    if strategy == "majority":
        counts = s.value_counts()
        return counts.index[0]

    raise ValueError(f"Unsupported label strategy: {strategy}")


def create_windows(
    df,
    feature_cols,
    window_size,
    step_size,
    label_col="label",
    run_col="run_id",
    tcol="t_rel",
    gap_threshold=0.025,
    min_samples=None,
    label_strategy="majority",
    include_series=False,
):
    """Create sliding windows within continuous segments only."""
    if min_samples is None:
        min_samples = window_size

    required = [run_col, tcol, label_col] + list(feature_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for windowing: {missing}")

    if step_size <= 0 or window_size <= 0:
        raise ValueError("window_size and step_size must be positive integers")

    segments = detect_continuous_segments(df, run_col=run_col, tcol=tcol, gap_threshold=gap_threshold)
    rows = []

    for seg in segments:
        seg_df = df.loc[seg["indices"]].sort_values(tcol).copy()
        if len(seg_df) < min_samples:
            continue

        n = len(seg_df)
        for start in range(0, n - window_size + 1, step_size):
            stop = start + window_size
            w = seg_df.iloc[start:stop]
            w_label = assign_window_labels(w[label_col], strategy=label_strategy)
            if pd.isna(w_label):
                continue

            row = {
                "run_id": seg["run_id"],
                "segment_id": seg["segment_id"],
                "window_start_idx": int(start),
                "window_end_idx": int(stop - 1),
                "t_start": float(w[tcol].iloc[0]),
                "t_end": float(w[tcol].iloc[-1]),
                "n_samples": int(window_size),
                "label": w_label,
            }

            if include_series:
                for col in feature_cols:
                    row[f"{col}__series"] = w[col].to_numpy(dtype=float)

            rows.append(row)

    return pd.DataFrame(rows)


def save_windowed_run(windows_df, output_path):
    """Save window metadata/features as a parquet or CSV file based on suffix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        windows_df.to_parquet(output_path, index=False)
    else:
        windows_df.to_csv(output_path, index=False)

    return output_path
