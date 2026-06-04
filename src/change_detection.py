"""
Online CUSUM transition detector for terrain classification.

This is the §6.4 (Blanke et al., 2016) version of the transition analysis.
Unlike the n_stable rule in src/transition.py, the detector here:
  - decides ONLINE whether a change happened (no oracle target label)
  - declares which class it switched TO (detection + isolation)
  - works on classifier soft posteriors, not hard argmax labels
  - has a single principled threshold h tied to a target false-alarm rate

Output schema mirrors analyze_transitions() so compare_models(),
plot_transition_timeline(), and plot_detection_delay_histogram() in
src/transition.py keep working unchanged. New columns/keys (false_alarm
rate, wrong_class outcome) are added on top.

Equation references in comments are to:
    Blanke, Kinnaert, Lunze, Staroswiecki (2016).
    "Diagnosis and Fault-Tolerant Control", 3rd ed., Springer (Chapter 6.4).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# Core: bank-of-CUSUMs detector

def cusum_bank(
    proba: np.ndarray,
    classes: list[str],
    h: float,
    c0_init: str | int | None = None,
    eps: float = 1e-9,
    warmup: int = 0,
) -> list[dict]:
    """Run a bank of CUSUMs over a posterior stream and emit alarm events.

    For each candidate post-change class c != c0 we accumulate the per-step
    log-likelihood ratio (Eq. 6.62 in Blanke et al.):

        s_c(k) = ln( p_c(k) / p_{c0}(k) )
        g_c(k) = max(0, g_c(k-1) + s_c(k))           # Eq. 6.91 recursive form

    An alarm fires when max_c g_c(k) > h. The declared class is the argmax
    CUSUM; the change-time estimate \\hat{k}_0 is recovered from Eq. 6.63 as
    the argmin of the cumulative sum S_c(j) for j <= k_a, then state resets.

    Returns a list of dicts: {'k_a': int, 'c_hat': str, 'k0_hat': int, 'g_max': float}.
    """
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2-D, got shape {proba.shape}")
    T, C = proba.shape
    if C != len(classes):
        raise ValueError(f"classes has {len(classes)} names but proba has {C} cols")
    if T == 0:
        return []

    if c0_init is None:
        c0_idx = int(np.argmax(proba[0]))
    elif isinstance(c0_init, str):
        c0_idx = classes.index(c0_init)
    else:
        c0_idx = int(c0_init)

    log_p = np.log(np.clip(proba, eps, 1.0))

    # State for the bank — one CUSUM per non-active class. Keep all C slots
    # for indexing convenience; the active one is zeroed at every step.
    g = np.zeros(C, dtype=float)
    S = np.zeros(C, dtype=float)
    # Per-class running min(S_c) and the j at which it occurred — used for
    # the change-time estimate \\hat{k}_0 (Eq. 6.63).
    S_min = np.zeros(C, dtype=float)
    S_argmin = np.zeros(C, dtype=int)

    alarms: list[dict] = []

    for k in range(T):
        s = log_p[k] - log_p[k, c0_idx]
        g = np.maximum(0.0, g + s)
        S = S + s
        update_min = S < S_min
        S_min = np.where(update_min, S, S_min)
        S_argmin = np.where(update_min, k, S_argmin)
        # The active class's own CUSUM is meaningless (s_{c0} ≡ 0)
        g[c0_idx] = 0.0

        if k < warmup:
            continue

        c_hat_idx = int(np.argmax(g))
        if g[c_hat_idx] > h:
            k0_hat = int(S_argmin[c_hat_idx])
            alarms.append({
                "k_a": k,
                "c_hat": classes[c_hat_idx],
                "k0_hat": k0_hat,
                "g_max": float(g[c_hat_idx]),
            })
            # Reinitialise: switch active class and reset all state
            c0_idx = c_hat_idx
            g[:] = 0.0
            S[:] = 0.0
            S_min[:] = 0.0
            S_argmin[:] = k

    return alarms


# Threshold calibration

def calibrate_threshold(
    proba_train: np.ndarray,
    labels_train: np.ndarray,
    classes: list[str],
    target_mtbfa_sec: float,
    step_sec: float = 1.0,
    h_grid: np.ndarray | None = None,
    min_segment_windows: int = 5,
) -> dict:
    """Pick h so that empirical mean time between alarms on stationary segments ≈ target_mtbfa_sec.

    Stationary segment = consecutive windows with the same true label. We run
    cusum_bank on each segment independently, sum alarms, and divide total
    stationary time by total alarms. We pick the smallest h whose MTBFA meets
    the target — Algorithm 6.6 in Blanke et al.

    Returns: {'h', 'h_grid', 'mtbfa_grid', 'alarms_grid', 'stationary_sec'}.
    """
    if h_grid is None:
        h_grid = np.linspace(1.0, 50.0, 50)

    labels = np.asarray(labels_train).astype(str)
    n = len(labels)
    if n != len(proba_train):
        raise ValueError("labels_train length must match proba_train rows")

    boundaries = [0]
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            boundaries.append(i)
    boundaries.append(n)
    segments = [(boundaries[i], boundaries[i + 1])
                for i in range(len(boundaries) - 1)]
    segments = [(s, e) for s, e in segments
                if e - s >= min_segment_windows]

    if not segments:
        raise ValueError("No stationary segments long enough to calibrate h")

    stationary_sec = sum((e - s) for s, e in segments) * step_sec

    alarms_per_h: list[int] = []
    mtbfa_per_h: list[float] = []
    for h in h_grid:
        n_alarms = 0
        for s, e in segments:
            seg_proba = proba_train[s:e]
            seg_label = labels[s]
            # Initialise active class to the segment's true label — otherwise
            # the bank would 'discover' the label at the segment start.
            seg_alarms = cusum_bank(
                seg_proba, classes, h=h, c0_init=seg_label,
            )
            n_alarms += len(seg_alarms)
        alarms_per_h.append(n_alarms)
        mtbfa_per_h.append(stationary_sec / n_alarms if n_alarms > 0 else np.inf)

    mtbfa_arr = np.array(mtbfa_per_h)
    # Pick the smallest h whose empirical MTBFA >= target; fall back to largest h if none meets.
    meeting = np.where(mtbfa_arr >= target_mtbfa_sec)[0]
    if len(meeting) == 0:
        chosen_idx = int(np.argmax(h_grid))
    else:
        chosen_idx = int(meeting[0])

    return {
        "h": float(h_grid[chosen_idx]),
        "h_grid": h_grid,
        "mtbfa_grid": mtbfa_arr,
        "alarms_grid": np.array(alarms_per_h),
        "stationary_sec": float(stationary_sec),
    }


# Alarm ↔ GT matching

def match_alarms_to_gt(
    alarms: list[dict],
    gt_transitions: list[dict],
    n_windows: int,
    times: np.ndarray,
    run_id: str = "unknown",
    grace_windows: int = 1,
    n_stable: int = 3,
    threshold_sec: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair detector alarms with ground-truth transitions.

    Pairing rule per the report's "Evaluation against GT" section:
        For each GT transition (t_k, ℓ_k) with bound (t_k - grace, t_{k+1}):
          - first alarm in window with c_hat == ℓ_k → 'detected'
          - first alarm in window with c_hat != ℓ_k → 'wrong_class'
          - no alarm in window                     → 'missed'
          - too few windows remain after t_k       → 'boundary'
        Any alarm NOT consumed by a GT pairing is logged as a 'false_alarm'.

    Returns (transitions_df, false_alarms_df) with same columns as analyze_transitions plus g_max.
    """
    rows: list[dict] = []
    used_alarm_idx: set[int] = set()
    times = np.asarray(times)

    alarm_ks = np.array([a["k_a"] for a in alarms], dtype=int) if alarms else np.array([], dtype=int)

    for k_idx, trans in enumerate(gt_transitions):
        t_idx = trans["transition_idx"]
        next_t_idx = (gt_transitions[k_idx + 1]["transition_idx"]
                      if k_idx + 1 < len(gt_transitions) else n_windows)
        target = trans["to_label"]
        gt_time = trans["gt_time"]

        # Boundary: mirror analyze_transitions() so detectors are comparable
        if t_idx + n_stable > n_windows:
            rows.append({
                "run_id": run_id,
                "transition_idx": t_idx,
                "from_label": trans["from_label"],
                "to_label": target,
                "gt_time": gt_time,
                "detection_time": float("nan"),
                "detection_delay_sec": float("nan"),
                "outcome": "boundary",
                "is_late": False,
                "n_stable_used": n_stable,
                "g_max": float("nan"),
            })
            continue

        lo = t_idx - grace_windows
        hi = next_t_idx
        candidates = [i for i, k in enumerate(alarm_ks)
                      if lo <= k < hi and i not in used_alarm_idx]

        if not candidates:
            rows.append({
                "run_id": run_id,
                "transition_idx": t_idx,
                "from_label": trans["from_label"],
                "to_label": target,
                "gt_time": gt_time,
                "detection_time": float("nan"),
                "detection_delay_sec": float("nan"),
                "outcome": "missed",
                "is_late": False,
                "n_stable_used": n_stable,
                "g_max": float("nan"),
            })
            continue

        first_i = min(candidates, key=lambda i: alarm_ks[i])
        used_alarm_idx.add(first_i)
        a = alarms[first_i]
        det_time = float(times[a["k_a"]])
        # Delay can be negative if alarm fires inside grace window before t_k — clamp to 0
        delay = max(0.0, det_time - gt_time)
        outcome = "detected" if a["c_hat"] == target else "wrong_class"

        rows.append({
            "run_id": run_id,
            "transition_idx": t_idx,
            "from_label": trans["from_label"],
            "to_label": target,
            "gt_time": gt_time,
            "detection_time": det_time,
            "detection_delay_sec": delay,
            "outcome": outcome,
            "is_late": delay > threshold_sec,
            "n_stable_used": n_stable,
            "g_max": a["g_max"],
        })

    transitions_df = pd.DataFrame(rows, columns=[
        "run_id", "transition_idx", "from_label", "to_label", "gt_time",
        "detection_time", "detection_delay_sec", "outcome", "is_late",
        "n_stable_used", "g_max",
    ])

    fa_rows = []
    for i, a in enumerate(alarms):
        if i in used_alarm_idx:
            continue
        fa_rows.append({
            "run_id": run_id,
            "k_a": a["k_a"],
            "alarm_time": float(times[a["k_a"]]) if a["k_a"] < len(times) else float("nan"),
            "c_hat": a["c_hat"],
            "g_max": a["g_max"],
        })
    false_alarms_df = pd.DataFrame(fa_rows, columns=[
        "run_id", "k_a", "alarm_time", "c_hat", "g_max",
    ])

    return transitions_df, false_alarms_df


# Drop-in replacement for analyze_transitions()

def analyze_transitions_cusum(
    df: pd.DataFrame,
    proba: np.ndarray,
    classes: list[str],
    h: float,
    time_col: str = "time",
    true_col: str = "true_label",
    grace_windows: int = 1,
    n_stable: int = 3,
    threshold_sec: float = 10.0,
    warmup: int = 0,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Per-run CUSUM analysis with same call signature spirit as analyze_transitions().

    Returns (transitions_df, summary, false_alarms_df) with the same columns/keys
    as analyze_transitions(), plus extras for wrong-class and false-alarm counts.
    """
    # Local import to avoid a circular dep at module load
    from src.transition import detect_gt_transitions

    if len(proba) != len(df):
        raise ValueError(
            f"proba has {len(proba)} rows but df has {len(df)} — they must align."
        )

    run_id = df["run_id"].iloc[0] if "run_id" in df.columns else "unknown"
    times = df[time_col].to_numpy()

    # Init active class to run's first true label so CUSUM doesn't fire a spurious "discovery" at t=0
    init_label = str(df[true_col].iloc[0])
    if init_label not in classes:
        init_label = None

    alarms = cusum_bank(proba, classes, h=h, c0_init=init_label, warmup=warmup)
    gt_transitions = detect_gt_transitions(df, time_col=time_col, true_col=true_col)

    transitions_df, false_alarms_df = match_alarms_to_gt(
        alarms=alarms,
        gt_transitions=gt_transitions,
        n_windows=len(df),
        times=times,
        run_id=str(run_id),
        grace_windows=grace_windows,
        n_stable=n_stable,
        threshold_sec=threshold_sec,
    )

    n_total = len(transitions_df)
    n_detected = int((transitions_df["outcome"] == "detected").sum())
    n_wrong = int((transitions_df["outcome"] == "wrong_class").sum())
    n_missed = int((transitions_df["outcome"] == "missed").sum())
    n_boundary = int((transitions_df["outcome"] == "boundary").sum())
    effective = n_total - n_boundary

    det_delays = transitions_df.loc[
        transitions_df["outcome"] == "detected", "detection_delay_sec"
    ]

    # False-alarm rate per minute using total run duration as exposure proxy
    run_duration_sec = float(times[-1] - times[0]) if len(times) > 1 else float("nan")
    fa_per_min = (len(false_alarms_df) / (run_duration_sec / 60.0)
                  if run_duration_sec and run_duration_sec > 0 else float("nan"))

    summary: dict[str, Any] = {
        "n_transitions": n_total,
        "n_detected": n_detected,
        "n_wrong_class": n_wrong,
        "n_missed": n_missed,
        "n_boundary": n_boundary,
        "detection_rate": n_detected / effective if effective > 0 else float("nan"),
        "wrong_class_rate": n_wrong / effective if effective > 0 else float("nan"),
        "mean_delay_sec": float(det_delays.mean()) if len(det_delays) else float("nan"),
        "median_delay_sec": float(det_delays.median()) if len(det_delays) else float("nan"),
        "std_delay_sec": float(det_delays.std()) if len(det_delays) else float("nan"),
        "max_delay_sec": float(det_delays.max()) if len(det_delays) else float("nan"),
        "n_late": int(transitions_df.get("is_late", pd.Series(dtype=bool)).sum()),
        "n_false_alarms": len(false_alarms_df),
        "false_alarm_rate_per_min": fa_per_min,
        "h": float(h),
    }

    return transitions_df, summary, false_alarms_df


def compare_models_cusum(
    df: pd.DataFrame,
    proba_dict: dict[str, tuple[np.ndarray, list[str]]],
    h_dict: dict[str, float],
    time_col: str = "t_start",
    true_col: str = "label",
    run_col: str = "run_id",
    grace_windows: int = 1,
    n_stable: int = 3,
    threshold_sec: float = 10.0,
    warmup: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-model multi-run CUSUM wrapper. Output is column-compatible with compare_models().

    Args:
        proba_dict : {model_name: (proba_full, classes)} — proba_full row-aligned to df.index.
        h_dict     : {model_name: h} — per-model threshold (posteriors are sharper for some models).
    """
    run_ids = sorted(df[run_col].unique().tolist())
    all_trans: list[pd.DataFrame] = []
    all_fa: list[pd.DataFrame] = []

    for model_name, (proba_full, classes) in proba_dict.items():
        if len(proba_full) != len(df):
            raise ValueError(
                f"{model_name}: proba has {len(proba_full)} rows, df has {len(df)}"
            )
        h = h_dict[model_name]

        for run_id in run_ids:
            run_mask = df[run_col].astype(str) == str(run_id)
            run_pos = np.where(run_mask.to_numpy())[0]
            time_order = np.argsort(df.iloc[run_pos][time_col].to_numpy())
            sorted_pos = run_pos[time_order]

            run_df = (
                df.iloc[sorted_pos]
                .assign(
                    time=lambda x: x[time_col],
                    true_label=lambda x: x[true_col],
                )
                .reset_index(drop=True)
            )
            run_proba = proba_full[sorted_pos]

            t_df, _, fa_df = analyze_transitions_cusum(
                df=run_df,
                proba=run_proba,
                classes=classes,
                h=h,
                time_col="time",
                true_col="true_label",
                grace_windows=grace_windows,
                n_stable=n_stable,
                threshold_sec=threshold_sec,
                warmup=warmup,
            )
            if not t_df.empty:
                t_df.insert(0, "model", model_name)
                all_trans.append(t_df)
            if not fa_df.empty:
                fa_df.insert(0, "model", model_name)
                all_fa.append(fa_df)

    combined = (pd.concat(all_trans, ignore_index=True) if all_trans
                else pd.DataFrame(columns=[
                    "model", "run_id", "transition_idx", "from_label",
                    "to_label", "gt_time", "detection_time",
                    "detection_delay_sec", "outcome", "is_late",
                    "n_stable_used", "g_max",
                ]))
    combined_fa = (pd.concat(all_fa, ignore_index=True) if all_fa
                   else pd.DataFrame(columns=[
                       "model", "run_id", "k_a", "alarm_time", "c_hat", "g_max",
                   ]))
    return combined, combined_fa


# Operating-curve sweep

def operating_curve(
    df: pd.DataFrame,
    proba_full: np.ndarray,
    classes: list[str],
    h_grid: np.ndarray,
    time_col: str = "t_start",
    true_col: str = "label",
    run_col: str = "run_id",
    grace_windows: int = 1,
    n_stable: int = 3,
    threshold_sec: float = 10.0,
) -> pd.DataFrame:
    """Sweep h and report (mean delay, false-alarm rate, detection rate) per value."""
    rows = []
    fake = {"_": (proba_full, classes)}
    for h in h_grid:
        trans, fa = compare_models_cusum(
            df=df,
            proba_dict=fake,
            h_dict={"_": float(h)},
            time_col=time_col,
            true_col=true_col,
            run_col=run_col,
            grace_windows=grace_windows,
            n_stable=n_stable,
            threshold_sec=threshold_sec,
        )

        n_total = len(trans)
        n_det = int((trans["outcome"] == "detected").sum())
        n_wrong = int((trans["outcome"] == "wrong_class").sum())
        n_miss = int((trans["outcome"] == "missed").sum())
        n_bnd = int((trans["outcome"] == "boundary").sum())
        eff = n_total - n_bnd
        delays = trans.loc[trans["outcome"] == "detected", "detection_delay_sec"]

        # Total run time across all runs — used as exposure for false-alarm rate
        total_sec = 0.0
        for run_id in df[run_col].unique():
            t = df.loc[df[run_col] == run_id, time_col].to_numpy()
            if len(t) > 1:
                total_sec += float(t.max() - t.min())
        fa_per_min = (len(fa) / (total_sec / 60.0)
                      if total_sec > 0 else float("nan"))

        rows.append({
            "h": float(h),
            "n_alarms": n_det + n_wrong + len(fa),
            "n_detected": n_det,
            "n_wrong": n_wrong,
            "n_missed": n_miss,
            "n_boundary": n_bnd,
            "n_false_alarms": len(fa),
            "detection_rate": n_det / eff if eff > 0 else float("nan"),
            "wrong_class_rate": n_wrong / eff if eff > 0 else float("nan"),
            "mean_delay_sec": float(delays.mean()) if len(delays) else float("nan"),
            "median_delay_sec": float(delays.median()) if len(delays) else float("nan"),
            "false_alarm_rate_per_min": fa_per_min,
        })

    return pd.DataFrame(rows)
