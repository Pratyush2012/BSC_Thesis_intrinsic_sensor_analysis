"""
src/change_detection.py
-----------------------
Online CUSUM transition detector for terrain classification.

This is the §6.4 (Blanke et al., 2016) version of the transition analysis.
Unlike the n_stable rule in src/transition.py, the detector here:
  - decides ONLINE whether a change happened (no oracle target label)
  - declares which class it switched TO (detection + isolation)
  - works on classifier soft posteriors, not hard argmax labels
  - has a single principled threshold h tied to a target false-alarm rate

The output schema mirrors analyze_transitions() so compare_models(),
plot_transition_timeline(), and plot_detection_delay_histogram() in
src/transition.py keep working unchanged. New columns/keys (false_alarm
rate, wrong_class outcome) are added on top.

Equation references in comments are to:
    Blanke, Kinnaert, Lunze, Staroswiecki (2016).
    "Diagnosis and Fault-Tolerant Control", 3rd ed., Springer.
The PDF I worked from has change-detection in Chapter 6.4, so eq. numbers
look like (6.62), (6.91), etc. — same content as §7.2 in the ToC of the
front-matter PDF.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ===========================================================================
# CORE: BANK-OF-CUSUMS DETECTOR
# ===========================================================================

def cusum_bank(
    proba: np.ndarray,
    classes: list[str],
    h: float,
    c0_init: str | int | None = None,
    eps: float = 1e-9,
    warmup: int = 0,
) -> list[dict]:
    """
    Run a bank of CUSUMs over a posterior stream and emit alarm events.

    For each candidate post-change class c != c0 we accumulate the per-step
    log-likelihood ratio (Eq. 6.62 in Blanke et al.):

        s_c(k) = ln( p_c(k) / p_{c0}(k) )
        g_c(k) = max(0, g_c(k-1) + s_c(k))           # Eq. 6.91 recursive form

    An alarm fires the first time max_c g_c(k) > h. The declared class is
    the argmax CUSUM, the change-time estimate \hat{k}_0 is recovered from
    Eq. 6.63 as the argmin of the cumulative sum S_c(j) for j <= k_a, and
    everything is reset to zero (re-init step of Algorithm 6.3).

    Args:
        proba    : (T, C) posterior matrix, time-sorted, rows sum to ~1.
        classes  : list of length C with class names (the column order of proba).
        h        : CUSUM threshold. Pick via calibrate_threshold().
        c0_init  : initial active class. If None, use argmax of the first
                   non-zero row. If a string it must be in `classes`; if int
                   it is treated as a column index.
        eps      : floor for log to avoid log(0) when a class has zero proba.
        warmup   : number of windows to skip at the start of the stream
                   (no alarm allowed). Useful when the classifier emits
                   garbage right at run start.

    Returns:
        List of dicts, one per alarm:
            {'k_a': int, 'c_hat': str, 'k0_hat': int, 'g_max': float}
        sorted by k_a ascending.
    """
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2-D, got shape {proba.shape}")
    T, C = proba.shape
    if C != len(classes):
        raise ValueError(f"classes has {len(classes)} names but proba has {C} cols")
    if T == 0:
        return []

    # Resolve initial active class index
    if c0_init is None:
        # Use the first row's argmax as the starting hypothesis
        c0_idx = int(np.argmax(proba[0]))
    elif isinstance(c0_init, str):
        c0_idx = classes.index(c0_init)
    else:
        c0_idx = int(c0_init)

    log_p = np.log(np.clip(proba, eps, 1.0))   # (T, C)

    # State for the bank — one CUSUM per non-active class. I keep all C slots
    # for indexing convenience and just zero out the active one.
    g = np.zeros(C, dtype=float)
    S = np.zeros(C, dtype=float)
    # Per-class running min(S_c) and the j at which it occurred — used for
    # the change-time estimate \hat{k}_0 (Eq. 6.63).
    S_min = np.zeros(C, dtype=float)
    S_argmin = np.zeros(C, dtype=int)

    alarms: list[dict] = []

    for k in range(T):
        # Increment s_c(k) = log p_c - log p_{c0} for every c != c0
        s = log_p[k] - log_p[k, c0_idx]
        g = np.maximum(0.0, g + s)
        S = S + s
        # Track min(S_c) up to time k for change-time recovery
        update_min = S < S_min
        S_min = np.where(update_min, S, S_min)
        S_argmin = np.where(update_min, k, S_argmin)
        # The active class's own CUSUM is meaningless (s_{c0} ≡ 0); ignore it
        g[c0_idx] = 0.0

        if k < warmup:
            continue

        # Alarm test: max over candidate classes
        c_hat_idx = int(np.argmax(g))
        if g[c_hat_idx] > h:
            # \hat{k}_0 = argmin_{1<=j<=k_a} S_{c_hat}(j) — the time the
            # cumulative sum for the winning class last hit its minimum
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


# ===========================================================================
# THRESHOLD CALIBRATION
# ===========================================================================

def calibrate_threshold(
    proba_train: np.ndarray,
    labels_train: np.ndarray,
    classes: list[str],
    target_mtbfa_sec: float,
    step_sec: float = 1.0,
    h_grid: np.ndarray | None = None,
    min_segment_windows: int = 5,
) -> dict:
    """
    Pick h so that on stationary segments of the training stream, the
    empirical mean time between alarms ≈ target_mtbfa_sec.

    A 'stationary segment' = consecutive windows with the same true label
    (between successive GT transitions). We run cusum_bank on each segment
    independently, count alarms, and divide total stationary time by total
    alarms to get the empirical MTBFA. We then pick the h on the grid whose
    MTBFA is closest to (and >=) the target — i.e. the loosest h that still
    meets the false-alarm budget.

    This corresponds to step "Choose h to meet specified mean time between
    false alarms" of Algorithm 6.6 in Blanke et al.

    Args:
        proba_train         : (T, C) posteriors, concatenated training stream.
        labels_train        : (T,) true class names, same order as proba_train.
        classes             : column order of proba_train.
        target_mtbfa_sec    : desired mean time between false alarms (seconds).
        step_sec            : seconds per window (window step). 1.0 by default.
        h_grid              : array of h values to try. Default: np.linspace(1, 50, 50).
        min_segment_windows : skip stationary segments shorter than this
                              (too short for any alarm to fire anyway).

    Returns:
        Dict with keys:
            'h'              : chosen threshold
            'h_grid'         : array of tried h values
            'mtbfa_grid'     : empirical MTBFA per h (seconds; np.inf if zero alarms)
            'alarms_grid'    : alarms per h (int)
            'stationary_sec' : total stationary time used (seconds)
    """
    if h_grid is None:
        h_grid = np.linspace(1.0, 50.0, 50)

    # Slice the training stream into stationary segments
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
            # Initialise the active class to the segment's true label —
            # otherwise the very first alarm at the segment start would be
            # spurious (the bank would 'discover' the label).
            seg_alarms = cusum_bank(
                seg_proba, classes, h=h, c0_init=seg_label,
            )
            n_alarms += len(seg_alarms)
        alarms_per_h.append(n_alarms)
        mtbfa_per_h.append(stationary_sec / n_alarms if n_alarms > 0 else np.inf)

    mtbfa_arr = np.array(mtbfa_per_h)
    # Pick the smallest h whose empirical MTBFA >= target. If none meets,
    # fall back to the largest h (most conservative).
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


# ===========================================================================
# ALARM ↔ GT MATCHING
# ===========================================================================

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
    """
    Pair detector alarms with ground-truth transitions and produce a
    transitions_df with the same columns as analyze_transitions().

    Pairing rule (per the report's "Evaluation against GT" section):
        For each GT transition (t_k, ℓ_k) with bound (t_k - grace, t_{k+1}):
          - first alarm in that window with c_hat == ℓ_k → 'detected'
          - first alarm in that window with c_hat != ℓ_k → 'wrong_class'
          - no alarm in that window                      → 'missed'
          - too few windows remain after t_k             → 'boundary'
        Any alarm NOT consumed by a GT pairing is logged as a 'false_alarm'.

    Args:
        alarms         : output of cusum_bank() — list of {k_a, c_hat, k0_hat, g_max}
        gt_transitions : output of detect_gt_transitions() for this run
        n_windows      : len(run_df), used for the boundary check
        times          : (n_windows,) array of t_start per window — gives
                         absolute seconds for delay computation
        run_id         : tagged onto every output row
        grace_windows  : how many windows BEFORE t_k still count as the same
                         transition (CUSUM can fire on the last pre-change
                         window if the change is sharp). 1 by default.
        n_stable      : carried through to the boundary check, mirroring
                         analyze_transitions(), so the comparison is fair.
        threshold_sec  : delays > this are flagged is_late=True.

    Returns:
        (transitions_df, false_alarms_df)

        transitions_df columns (same as analyze_transitions, plus 'g_max'):
            run_id, transition_idx, from_label, to_label, gt_time,
            detection_time, detection_delay_sec, outcome, is_late, n_stable_used, g_max

        false_alarms_df columns:
            run_id, k_a, alarm_time, c_hat, g_max
    """
    rows: list[dict] = []
    used_alarm_idx: set[int] = set()
    times = np.asarray(times)

    # Pre-compute alarm k_a indices for fast windowed lookup
    alarm_ks = np.array([a["k_a"] for a in alarms], dtype=int) if alarms else np.array([], dtype=int)

    for k_idx, trans in enumerate(gt_transitions):
        t_idx = trans["transition_idx"]
        next_t_idx = (gt_transitions[k_idx + 1]["transition_idx"]
                      if k_idx + 1 < len(gt_transitions) else n_windows)
        target = trans["to_label"]
        gt_time = trans["gt_time"]

        # Boundary: not enough windows remain to observe a meaningful detection.
        # Mirror analyze_transitions() so the two detectors are comparable.
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

        # Find first alarm in [t_idx - grace, next_t_idx)
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

        # Pick the earliest candidate (by k_a)
        first_i = min(candidates, key=lambda i: alarm_ks[i])
        used_alarm_idx.add(first_i)
        a = alarms[first_i]
        det_time = float(times[a["k_a"]])
        # Delay can be negative if the alarm fires inside the grace window
        # before t_k — clamp to 0 so the metric stays interpretable.
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

    # Anything not consumed = false alarm
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


# ===========================================================================
# DROP-IN REPLACEMENT FOR analyze_transitions()
# ===========================================================================

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
    """
    Same call signature spirit as analyze_transitions() but using CUSUM.

    This is the function the notebook calls per run. It:
      1. Detects GT transitions exactly like analyze_transitions().
      2. Runs cusum_bank() on the run's proba slice.
      3. Matches alarms to GT transitions.
      4. Returns (transitions_df, summary, false_alarms_df) with the same
         columns/keys as analyze_transitions(), plus extra ones for
         wrong-class and false-alarm counts.

    Args:
        df            : single-run, time-sorted DataFrame with [time_col, true_col].
        proba         : (T, C) posterior matrix for THIS run's rows of df,
                         in the same row order as df.
        classes       : column names of proba.
        h             : CUSUM threshold (from calibrate_threshold()).
        grace_windows : passed to match_alarms_to_gt.
        n_stable      : only used for the boundary rule (kept for parity
                         with analyze_transitions()).
        threshold_sec : delays > this flagged is_late=True.
        warmup        : passed to cusum_bank() — no alarms in the first
                         `warmup` windows.

    Returns:
        transitions_df, summary, false_alarms_df
    """
    # Local import to avoid a circular dep at module load
    from src.transition import detect_gt_transitions

    if len(proba) != len(df):
        raise ValueError(
            f"proba has {len(proba)} rows but df has {len(df)} — they must align."
        )

    run_id = df["run_id"].iloc[0] if "run_id" in df.columns else "unknown"
    times = df[time_col].to_numpy()

    # Initialise active class to the run's first true label so CUSUM doesn't
    # fire a spurious "discovery" alarm at t=0.
    init_label = str(df[true_col].iloc[0])
    if init_label not in classes:
        init_label = None  # let cusum_bank fall back to argmax of row 0

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

    # Summary stats — same keys as analyze_transitions(), plus new ones
    n_total = len(transitions_df)
    n_detected = int((transitions_df["outcome"] == "detected").sum())
    n_wrong = int((transitions_df["outcome"] == "wrong_class").sum())
    n_missed = int((transitions_df["outcome"] == "missed").sum())
    n_boundary = int((transitions_df["outcome"] == "boundary").sum())
    effective = n_total - n_boundary

    det_delays = transitions_df.loc[
        transitions_df["outcome"] == "detected", "detection_delay_sec"
    ]

    # False-alarm rate in alarms / minute, using the run's stationary time
    # (everything not within grace of a GT transition). For simplicity I use
    # total run duration — close enough when the run has many GT transitions.
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
    """
    Multi-model multi-run wrapper that mirrors compare_models() but runs
    the CUSUM detector. Output transitions_df is column-compatible with
    compare_models() so plot_transition_timeline() and
    plot_detection_delay_histogram() work without any change.

    Args:
        df          : full feature DataFrame with run_col, time_col, true_col.
        proba_dict  : {model_name: (proba_full, classes)} — proba_full is the
                       (N_total_windows, C) matrix from run_*_loro_cv_proba(),
                       row-aligned to df.index.
        h_dict      : {model_name: h} — chosen threshold per model (same model
                       can have a different h since posteriors are sharper for
                       some models than others).

    Returns:
        (combined_transitions_df, combined_false_alarms_df)
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
            # Sort positions by time so proba rows line up with sorted df
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


# ===========================================================================
# OPERATING-CURVE SWEEP
# ===========================================================================

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
    """
    Sweep h and report (mean delay, false-alarm rate, detection rate) for
    each value. This is the §6.4 equivalent of the n_stable sweep already
    in notebook 12 — plot one curve per detector type to compare them
    directly on the same axes.

    Returns a DataFrame with one row per h value:
        h, n_alarms, n_detected, n_wrong, n_missed, n_boundary,
        n_false_alarms, detection_rate, wrong_class_rate,
        mean_delay_sec, median_delay_sec, false_alarm_rate_per_min
    """
    # Build one fake h_dict per call so we can reuse compare_models_cusum
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

        # Total run time across all runs — use it as the "exposure" for the
        # false-alarm rate
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
