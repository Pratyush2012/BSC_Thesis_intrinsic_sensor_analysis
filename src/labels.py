"""Label assignment utilities for sensor data."""

from dataclasses import dataclass
from typing import List, Optional, Dict, TYPE_CHECKING
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


if TYPE_CHECKING:
    from src.io import RunData


@dataclass
class LabelRange:
    """A single time range for a label."""
    start: float
    end: float


@dataclass
class LabelConfig:
    """Configuration for a single label with its time ranges."""
    name: str
    ranges: List[LabelRange]


def load_label_config(config_path: str) -> Dict[str, LabelConfig]:
    """Load label definitions from a JSON configuration file.

    JSON format: {"label_name": [[start, end], [start, end], ...], ...}
    """
    with open(config_path, 'r') as f:
        raw_config = json.load(f)

    label_configs = {}
    for label_name, time_ranges in raw_config.items():
        ranges = [LabelRange(start=r[0], end=r[1]) for r in time_ranges]
        label_configs[label_name] = LabelConfig(name=label_name, ranges=ranges)

    return label_configs


def assign_label(timestamp: float, label_configs: Dict[str, LabelConfig]) -> Optional[str]:
    """Return the label whose range contains timestamp, or None."""
    for label_name, config in label_configs.items():
        for label_range in config.ranges:
            if label_range.start <= timestamp <= label_range.end:
                return label_name
    return None


def label_dataframe(df: pd.DataFrame,
                   label_configs: Dict[str, LabelConfig],
                   time_column: str = 't') -> pd.DataFrame:
    """Add a 'label' column to a DataFrame based on Unix time ranges."""
    df_copy = df.copy()
    df_copy['label'] = df_copy[time_column].apply(
        lambda t: assign_label(t, label_configs)
    )
    return df_copy


def find_label_config(run_dir: Path) -> Optional[Path]:
    """Find the label config for a given run directory.

    Checks the run directory first, then the parent test directory.
    """
    run_dir = Path(run_dir)

    run_config = run_dir / "labels_config.json"
    if run_config.exists():
        return run_config

    test_config = run_dir.parent / "labels_config.json"
    if test_config.exists():
        return test_config

    return None


def discover_labeled_runs(raw_root: Path) -> Dict[Path, Path]:
    """Discover all runs with their respective label configurations."""
    raw_root = Path(raw_root)
    runs_with_labels = {}

    test_dirs = [d for d in raw_root.iterdir() if d.is_dir()]
    test_dirs.sort()

    for test_dir in test_dirs:
        run_dirs = [d for d in test_dir.glob("log_*") if d.is_dir()]
        for run_dir in run_dirs:
            label_config_path = find_label_config(run_dir)
            if label_config_path:
                runs_with_labels[run_dir] = label_config_path

    # Farm dataset: Run1/Run2 style directories with labels_config at that level
    for test_dir in test_dirs:
        for subdir in test_dir.iterdir():
            if subdir.is_dir() and (subdir / "labels_config.json").exists():
                sensor_files = list(subdir.glob("log_t0_*.txt")) + list(subdir.glob("log_t0_*.csv"))
                if sensor_files:
                    runs_with_labels[subdir] = subdir / "labels_config.json"

    return runs_with_labels


def label_run_sensors(run: 'RunData', label_configs: Dict[str, LabelConfig]) -> Dict[str, pd.DataFrame]:
    """Apply labels to all sensors in a run."""
    labeled_sensors = {}
    for sensor_name, sensor_df in run.sensors().items():
        labeled_sensors[sensor_name] = label_dataframe(sensor_df, label_configs, time_column='t')
    return labeled_sensors


def _add_label_backgrounds(
    ax,
    df: pd.DataFrame,
    tcol: str,
    label_colors: Dict,
    gap_factor: float = 3.0,
) -> None:
    """Add colored label spans while breaking regions across large time gaps."""
    if 'label' not in df.columns or tcol not in df.columns or df.empty:
        return

    df_with_idx = df[[tcol, 'label']].reset_index(drop=True).copy()

    dt = df_with_idx[tcol].diff().dropna()
    positive_dt = dt[dt > 0]
    gap_threshold = None
    if not positive_dt.empty:
        gap_threshold = float(positive_dt.median()) * gap_factor

    def close_segment(start_idx: int, end_idx: int, label_name) -> None:
        if start_idx is None or end_idx is None or label_name is None:
            return
        start_time = df_with_idx.loc[start_idx, tcol]
        end_time = df_with_idx.loc[end_idx, tcol]
        color = label_colors.get(label_name, '#eeeeee')
        ax.axvspan(start_time, end_time, alpha=0.3, color=color, zorder=0)

    current_label = None
    start_idx = None
    prev_time = None

    for idx in range(len(df_with_idx)):
        time_now = df_with_idx.loc[idx, tcol]
        label_now = df_with_idx.loc[idx, 'label']
        if pd.isna(label_now):
            label_now = None

        gap_break = False
        if prev_time is not None and gap_threshold is not None:
            gap_break = (time_now - prev_time) > gap_threshold

        if gap_break and current_label is not None and start_idx is not None:
            close_segment(start_idx, idx - 1, current_label)
            current_label = None
            start_idx = None

        if label_now != current_label:
            if current_label is not None and start_idx is not None:
                close_segment(start_idx, idx - 1, current_label)

            current_label = label_now
            start_idx = idx if current_label is not None else None

        prev_time = time_now

    if current_label is not None and start_idx is not None:
        close_segment(start_idx, len(df_with_idx) - 1, current_label)


def _build_label_color_map(dfs: list) -> Dict:
    """Return a {label: pastel_rgba} mapping built from all unique labels in *dfs*."""
    all_labels: set = set()
    for df in dfs:
        if isinstance(df, pd.DataFrame) and 'label' in df.columns:
            all_labels.update(df['label'].dropna().unique())

    if not all_labels:
        return {}

    cmap = matplotlib.colormaps['tab10']
    label_colors: Dict = {}
    for i, label in enumerate(sorted(all_labels)):
        rgba = cmap(i % 10)
        light_color = tuple(min(1.0, c + 0.4) if j < 3 else c for j, c in enumerate(rgba))
        label_colors[label] = light_color
    return label_colors


def validate_label_transitions(labeled_df: pd.DataFrame, max_transitions: int = 3) -> Dict:
    """Validate label transitions in a labeled DataFrame.

    Returns total transition count and sample rows around the first few transitions.
    """
    transitions = []
    for i in range(1, len(labeled_df)):
        if labeled_df['label'].iloc[i] != labeled_df['label'].iloc[i-1]:
            transitions.append(i)

    transition_samples = []
    for idx, transition_idx in enumerate(transitions[:max_transitions]):
        start = max(0, transition_idx - 2)
        end = min(len(labeled_df), transition_idx + 3)
        sample_rows = labeled_df[['t', 'label']].iloc[start:end]
        transition_samples.append({
            'transition_num': idx + 1,
            'row_idx': transition_idx,
            'samples': sample_rows
        })

    return {
        'total_transitions': len(transitions),
        'transition_samples': transition_samples
    }


def _add_label_legend(fig, all_labels, label_colors) -> None:
    if all_labels:
        legend_patches = [
            mpatches.Patch(color=label_colors[label], label=label, alpha=0.3)
            for label in sorted(all_labels)
        ]
        fig.legend(handles=legend_patches, loc='upper center',
                  ncol=min(len(legend_patches), 6), bbox_to_anchor=(0.5, 0.98),
                  title='Activity Labels')


def plot_labeled_sensors(
    acc: pd.DataFrame,
    gyro: pd.DataFrame,
    odo: pd.DataFrame,
    tcol: str = "t_rel",
    show: bool = False,
) -> plt.Figure:
    """Plot per-sensor panels (axes + magnitude) with color-coded labels for one run."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    label_colors = _build_label_color_map([acc, gyro, odo])
    all_labels = set(label_colors.keys())

    def add_label_backgrounds(ax, df, tcol):
        _add_label_backgrounds(ax=ax, df=df, tcol=tcol, label_colors=label_colors)

    add_label_backgrounds(axes[0, 0], acc, tcol)
    axes[0, 0].plot(acc[tcol], acc["ax"], label="ax", linewidth=0.8)
    axes[0, 0].plot(acc[tcol], acc["ay"], label="ay", linewidth=0.8)
    axes[0, 0].plot(acc[tcol], acc["az"], label="az", linewidth=0.8)
    axes[0, 0].set_title("Accelerometer axes (labeled)")
    axes[0, 0].set_ylabel("acc [g]")
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].legend()

    add_label_backgrounds(axes[0, 1], acc, tcol)
    acc_mag = np.sqrt(acc["ax"] ** 2 + acc["ay"] ** 2 + acc["az"] ** 2)
    axes[0, 1].plot(acc[tcol], acc_mag, linewidth=0.8)
    axes[0, 1].set_title("Accelerometer magnitude (labeled)")
    axes[0, 1].set_ylabel("|a| [g]")
    axes[0, 1].grid(True, alpha=0.4)

    add_label_backgrounds(axes[1, 0], gyro, tcol)
    axes[1, 0].plot(gyro[tcol], gyro["gx"], label="gx", linewidth=0.8)
    axes[1, 0].plot(gyro[tcol], gyro["gy"], label="gy", linewidth=0.8)
    axes[1, 0].plot(gyro[tcol], gyro["gz"], label="gz", linewidth=0.8)
    axes[1, 0].set_title("Gyroscope axes (labeled)")
    axes[1, 0].set_ylabel("gyro [deg/s]")
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].legend()

    add_label_backgrounds(axes[1, 1], gyro, tcol)
    gyro_mag = np.sqrt(gyro["gx"] ** 2 + gyro["gy"] ** 2 + gyro["gz"] ** 2)
    axes[1, 1].plot(gyro[tcol], gyro_mag, linewidth=0.8)
    axes[1, 1].set_title("Gyroscope magnitude (labeled)")
    axes[1, 1].set_ylabel("|g| [deg/s]")
    axes[1, 1].grid(True, alpha=0.4)

    add_label_backgrounds(axes[2, 0], odo, tcol)
    axes[2, 0].plot(odo[tcol], odo["v1"], label="v1", linewidth=0.8)
    axes[2, 0].plot(odo[tcol], odo["v2"], label="v2", linewidth=0.8)
    axes[2, 0].set_title("Odometry channels (labeled)")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_ylabel("velocity [m/s]")
    axes[2, 0].grid(True, alpha=0.4)
    axes[2, 0].legend()

    add_label_backgrounds(axes[2, 1], odo, tcol)
    speed = 0.5 * (odo["v1"] + odo["v2"])
    axes[2, 1].plot(odo[tcol], speed, linewidth=0.8)
    axes[2, 1].set_title("Odometry speed (labeled)")
    axes[2, 1].set_xlabel("time [s]")
    axes[2, 1].set_ylabel("speed [m/s]")
    axes[2, 1].grid(True, alpha=0.4)

    _add_label_legend(fig, all_labels, label_colors)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()

    return fig


def plot_labeled_acc(
    acc: pd.DataFrame,
    tcol: str = "t_rel",
    show: bool = False,
) -> plt.Figure:
    """Plot accelerometer data with individual axes and magnitude in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    label_colors = _build_label_color_map([acc])
    all_labels = set(label_colors.keys())

    def add_label_backgrounds(ax, df, tcol):
        _add_label_backgrounds(ax=ax, df=df, tcol=tcol, label_colors=label_colors)

    add_label_backgrounds(axes[0, 0], acc, tcol)
    axes[0, 0].plot(acc[tcol], acc["ax"], label="ax", linewidth=0.8, color='C0')
    axes[0, 0].set_title("Accelerometer - X Axis (ax)")
    axes[0, 0].set_ylabel("acc [g]")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].legend()

    add_label_backgrounds(axes[0, 1], acc, tcol)
    axes[0, 1].plot(acc[tcol], acc["ay"], label="ay", linewidth=0.8, color='C1')
    axes[0, 1].set_title("Accelerometer - Y Axis (ay)")
    axes[0, 1].set_ylabel("acc [g]")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].legend()

    add_label_backgrounds(axes[1, 0], acc, tcol)
    axes[1, 0].plot(acc[tcol], acc["az"], label="az", linewidth=0.8, color='C2')
    axes[1, 0].set_title("Accelerometer - Z Axis (az)")
    axes[1, 0].set_ylabel("acc [g]")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].legend()

    add_label_backgrounds(axes[1, 1], acc, tcol)
    acc_mag = np.sqrt(acc["ax"] ** 2 + acc["ay"] ** 2 + acc["az"] ** 2)
    axes[1, 1].plot(acc[tcol], acc_mag, linewidth=0.8, color='red')
    axes[1, 1].set_title("Accelerometer - Magnitude")
    axes[1, 1].set_ylabel("|a| [g]")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].grid(True, alpha=0.4)

    _add_label_legend(fig, all_labels, label_colors)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()

    return fig


def plot_labeled_gyro(
    gyro: pd.DataFrame,
    tcol: str = "t_rel",
    show: bool = False,
) -> plt.Figure:
    """Plot gyroscope data with individual axes and magnitude in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    label_colors = _build_label_color_map([gyro])
    all_labels = set(label_colors.keys())

    def add_label_backgrounds(ax, df, tcol):
        _add_label_backgrounds(ax=ax, df=df, tcol=tcol, label_colors=label_colors)

    add_label_backgrounds(axes[0, 0], gyro, tcol)
    axes[0, 0].plot(gyro[tcol], gyro["gx"], label="gx", linewidth=0.8, color='C0')
    axes[0, 0].set_title("Gyroscope - X Axis (gx)")
    axes[0, 0].set_ylabel("gyro [deg/s]")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].legend()

    add_label_backgrounds(axes[0, 1], gyro, tcol)
    axes[0, 1].plot(gyro[tcol], gyro["gy"], label="gy", linewidth=0.8, color='C1')
    axes[0, 1].set_title("Gyroscope - Y Axis (gy)")
    axes[0, 1].set_ylabel("gyro [deg/s]")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].legend()

    add_label_backgrounds(axes[1, 0], gyro, tcol)
    axes[1, 0].plot(gyro[tcol], gyro["gz"], label="gz", linewidth=0.8, color='C2')
    axes[1, 0].set_title("Gyroscope - Z Axis (gz)")
    axes[1, 0].set_ylabel("gyro [deg/s]")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].legend()

    add_label_backgrounds(axes[1, 1], gyro, tcol)
    gyro_mag = np.sqrt(gyro["gx"] ** 2 + gyro["gy"] ** 2 + gyro["gz"] ** 2)
    axes[1, 1].plot(gyro[tcol], gyro_mag, linewidth=0.8, color='red')
    axes[1, 1].set_title("Gyroscope - Magnitude")
    axes[1, 1].set_ylabel("|g| [deg/s]")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].grid(True, alpha=0.4)

    _add_label_legend(fig, all_labels, label_colors)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()

    return fig


def plot_labeled_odo(
    odo: pd.DataFrame,
    tcol: str = "t_rel",
    show: bool = False,
) -> plt.Figure:
    """Plot odometry data with individual channels and speed in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    label_colors = _build_label_color_map([odo])
    all_labels = set(label_colors.keys())

    def add_label_backgrounds(ax, df, tcol):
        _add_label_backgrounds(ax=ax, df=df, tcol=tcol, label_colors=label_colors)

    add_label_backgrounds(axes[0, 0], odo, tcol)
    axes[0, 0].plot(odo[tcol], odo["v1"], label="v1", linewidth=0.8, color='C0')
    axes[0, 0].set_title("Odometry - Left Wheel (v1)")
    axes[0, 0].set_ylabel("velocity [m/s]")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].legend()

    add_label_backgrounds(axes[0, 1], odo, tcol)
    axes[0, 1].plot(odo[tcol], odo["v2"], label="v2", linewidth=0.8, color='C1')
    axes[0, 1].set_title("Odometry - Right Wheel (v2)")
    axes[0, 1].set_ylabel("velocity [m/s]")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].legend()

    axes[1, 0].axis('off')

    add_label_backgrounds(axes[1, 1], odo, tcol)
    speed = 0.5 * (odo["v1"] + odo["v2"])
    axes[1, 1].plot(odo[tcol], speed, linewidth=0.8, color='red')
    axes[1, 1].set_title("Odometry - Speed (Average)")
    axes[1, 1].set_ylabel("speed [m/s]")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].grid(True, alpha=0.4)

    _add_label_legend(fig, all_labels, label_colors)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()

    return fig


def save_labeled_plot(
    fig: plt.Figure,
    run_id: str,
    output_root: Path,
    filename: str = "labeled_sensors_plot.png",
    dpi: int = 150
) -> Path:
    """Save a labeled sensor plot to disk under ``output_root/run_id/``."""
    output_root = Path(output_root)
    run_output_dir = output_root / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_output_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')

    return output_path
