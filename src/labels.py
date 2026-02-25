"""Label assignment utilities for sensor data."""

from dataclasses import dataclass
from typing import List, Optional, Dict, TYPE_CHECKING
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm


if TYPE_CHECKING:
    from src.io import RunData


@dataclass
class LabelRange:
    """A single time range for a label."""
    start: float  # Unix time
    end: float    # Unix time


@dataclass
class LabelConfig:
    """Configuration for a single label with its time ranges."""
    name: str
    ranges: List[LabelRange]


def load_label_config(config_path: str) -> Dict[str, LabelConfig]:
    """
    Load label definitions from a JSON configuration file.
    
    JSON format:
    {
        "label_name": [[start, end], [start, end], ...],
        ...
    }
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        Dictionary mapping label names to LabelConfig objects
    """
    with open(config_path, 'r') as f:
        raw_config = json.load(f)
    
    label_configs = {}
    for label_name, time_ranges in raw_config.items():
        ranges = [LabelRange(start=r[0], end=r[1]) for r in time_ranges]
        label_configs[label_name] = LabelConfig(name=label_name, ranges=ranges)
    
    return label_configs


def assign_label(timestamp: float, label_configs: Dict[str, LabelConfig]) -> Optional[str]:
    """
    Assign a label to a single timestamp.
    
    Args:
        timestamp: Unix time (seconds)
        label_configs: Dictionary of label configurations
        
    Returns:
        Label name if timestamp falls in a range, NaN otherwise
    """
    for label_name, config in label_configs.items():
        for label_range in config.ranges:
            if label_range.start <= timestamp <= label_range.end:
                return label_name
    return float('nan')  # Return NaN if no label matches


def label_dataframe(df: pd.DataFrame, 
                   label_configs: Dict[str, LabelConfig],
                   time_column: str = 't') -> pd.DataFrame:
    """
    Add a 'label' column to a DataFrame based on Unix time ranges.
    
    Args:
        df: DataFrame with a time column (Unix timestamp)
        label_configs: Dictionary of label configurations
        time_column: Name of the time column (default: 't')
        
    Returns:
        DataFrame with new 'label' column added
    """
    df_copy = df.copy()
    df_copy['label'] = df_copy[time_column].apply(
        lambda t: assign_label(t, label_configs)
    )
    return df_copy

def find_label_config(run_dir: Path) -> Optional[Path]:
    """
    Find the label config for a given run directory.
    
    Checks in the run directory first, then in the parent test directory.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Path to label config if found, None otherwise
    """
    run_dir = Path(run_dir)
    
    # Check for config in the run directory
    run_config = run_dir / "labels_config.json"
    if run_config.exists():
        return run_config
    
    # Check for config in the parent test directory
    test_config = run_dir.parent / "labels_config.json"
    if test_config.exists():
        return test_config
    
    return None


def discover_labeled_runs(raw_root: Path) -> Dict[Path, Path]:
    """
    Discover all runs with their respective label configurations.
    
    Finds all log_* directories under raw_root and maps them to their
    corresponding label configuration files.
    
    Args:
        raw_root: Root directory containing raw data (e.g., data/raw/)
        
    Returns:
        Dictionary mapping run directories to label config paths
    """
    raw_root = Path(raw_root)
    runs_with_labels = {}
    
    # Find all test directories
    test_dirs = [d for d in raw_root.iterdir() if d.is_dir()]
    test_dirs.sort()
    
    # For each test directory, find runs with label configs
    for test_dir in test_dirs:
        run_dirs = [d for d in test_dir.glob("log_*") if d.is_dir()]
        for run_dir in run_dirs:
            label_config_path = find_label_config(run_dir)
            if label_config_path:
                runs_with_labels[run_dir] = label_config_path
    
    return runs_with_labels


def label_run_sensors(run: 'RunData', label_configs: Dict[str, LabelConfig]) -> Dict[str, pd.DataFrame]:
    """
    Apply labels to all sensors in a run.
    
    Args:
        run: RunData object with sensor DataFrames
        label_configs: Dictionary of label configurations
        
    Returns:
        Dictionary mapping sensor names to labeled DataFrames
    """
    labeled_sensors = {}
    
    for sensor_name, sensor_df in run.sensors().items():
        labeled_sensors[sensor_name] = label_dataframe(sensor_df, label_configs, time_column='t')
    
    return labeled_sensors


def validate_label_transitions(labeled_df: pd.DataFrame, max_transitions: int = 3) -> Dict:
    """
    Validate label transitions in a labeled DataFrame.
    
    Args:
        labeled_df: DataFrame with 'label' column
        max_transitions: Maximum number of transitions to show in output
        
    Returns:
        Dictionary with transition info and sample rows
    """
    # Find label transitions
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


def plot_labeled_sensors(
    acc: pd.DataFrame,
    gyro: pd.DataFrame,
    odo: pd.DataFrame,
    tcol: str = "t_rel",
    show: bool = False,
) -> plt.Figure:
    """Plot per-sensor panels with color-coded labels for one run.
    
    Creates a 3x2 grid showing:
    - Left column: sensor axes (acc xyz, gyro xyz, odo v1/v2)
    - Right column: magnitudes (acc mag, gyro mag, odo speed)
    
    Label regions are highlighted with background colors.
    Colors are automatically assigned based on the unique labels present in the data.
    
    Args:
        acc: Labeled accelerometer DataFrame
        gyro: Labeled gyroscope DataFrame  
        odo: Labeled odometry DataFrame
        tcol: Time column name (default: 't_rel')
        show: Whether to display the plot immediately
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Find all unique labels across all sensors
    all_labels = set()
    for df in [acc, gyro, odo]:
        if 'label' in df.columns:
            unique_labels = df['label'].dropna().unique()
            all_labels.update(unique_labels)
    
    # Dynamically assign colors to labels using matplotlib's tab10 colormap
    # Convert to light pastel colors for background highlighting
    if all_labels:
        sorted_labels = sorted(all_labels)
        n_labels = len(sorted_labels)
        
        # Use tab10 for up to 10 labels, then cycle through if more
        cmap = cm.get_cmap('tab10')
        label_colors = {}
        for i, label in enumerate(sorted_labels):
            rgba = cmap(i % 10)
            # Convert to lighter pastel version (increase brightness)
            light_color = tuple(min(1.0, c + 0.4) if j < 3 else c for j, c in enumerate(rgba))
            label_colors[label] = light_color
    else:
        label_colors = {}
    
    # Helper function to add label backgrounds
    def add_label_backgrounds(ax, df, tcol):
        """Add colored background regions for each label."""
        if 'label' not in df.columns:
            return
        
        # Group consecutive rows with same label
        df_with_idx = df.reset_index(drop=True)
        current_label = None
        start_idx = None
        
        for idx in range(len(df_with_idx)):
            label = df_with_idx.loc[idx, 'label']
            
            # Convert NaN to None for comparison
            if pd.isna(label):
                label = None
                
            if label != current_label:
                # End previous region
                if current_label is not None and start_idx is not None:
                    start_time = df_with_idx.loc[start_idx, tcol]
                    end_time = df_with_idx.loc[idx - 1, tcol]
                    color = label_colors.get(current_label, '#eeeeee')
                    ax.axvspan(start_time, end_time, alpha=0.3, color=color, zorder=0)
                
                # Start new region
                current_label = label
                start_idx = idx
        
        # Handle last region
        if current_label is not None and start_idx is not None:
            start_time = df_with_idx.loc[start_idx, tcol]
            end_time = df_with_idx.loc[len(df_with_idx) - 1, tcol]
            color = label_colors.get(current_label, '#eeeeee')
            ax.axvspan(start_time, end_time, alpha=0.3, color=color, zorder=0)
    
    # Plot accelerometer axes
    add_label_backgrounds(axes[0, 0], acc, tcol)
    axes[0, 0].plot(acc[tcol], acc["ax"], label="ax", linewidth=0.8)
    axes[0, 0].plot(acc[tcol], acc["ay"], label="ay", linewidth=0.8)
    axes[0, 0].plot(acc[tcol], acc["az"], label="az", linewidth=0.8)
    axes[0, 0].set_title("Accelerometer axes (labeled)")
    axes[0, 0].set_ylabel("acc [g]")
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].legend()

    # Plot accelerometer magnitude
    add_label_backgrounds(axes[0, 1], acc, tcol)
    acc_mag = np.sqrt(acc["ax"] ** 2 + acc["ay"] ** 2 + acc["az"] ** 2)
    axes[0, 1].plot(acc[tcol], acc_mag, linewidth=0.8)
    axes[0, 1].set_title("Accelerometer magnitude (labeled)")
    axes[0, 1].set_ylabel("|a| [g]")
    axes[0, 1].grid(True, alpha=0.4)

    # Plot gyroscope axes
    add_label_backgrounds(axes[1, 0], gyro, tcol)
    axes[1, 0].plot(gyro[tcol], gyro["gx"], label="gx", linewidth=0.8)
    axes[1, 0].plot(gyro[tcol], gyro["gy"], label="gy", linewidth=0.8)
    axes[1, 0].plot(gyro[tcol], gyro["gz"], label="gz", linewidth=0.8)
    axes[1, 0].set_title("Gyroscope axes (labeled)")
    axes[1, 0].set_ylabel("gyro [deg/s]")
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].legend()

    # Plot gyroscope magnitude
    add_label_backgrounds(axes[1, 1], gyro, tcol)
    gyro_mag = np.sqrt(gyro["gx"] ** 2 + gyro["gy"] ** 2 + gyro["gz"] ** 2)
    axes[1, 1].plot(gyro[tcol], gyro_mag, linewidth=0.8)
    axes[1, 1].set_title("Gyroscope magnitude (labeled)")
    axes[1, 1].set_ylabel("|g| [deg/s]")
    axes[1, 1].grid(True, alpha=0.4)

    # Plot odometry channels
    add_label_backgrounds(axes[2, 0], odo, tcol)
    axes[2, 0].plot(odo[tcol], odo["v1"], label="v1", linewidth=0.8)
    axes[2, 0].plot(odo[tcol], odo["v2"], label="v2", linewidth=0.8)
    axes[2, 0].set_title("Odometry channels (labeled)")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_ylabel("velocity [m/s]")
    axes[2, 0].grid(True, alpha=0.4)
    axes[2, 0].legend()

    # Plot odometry speed
    add_label_backgrounds(axes[2, 1], odo, tcol)
    speed = 0.5 * (odo["v1"] + odo["v2"])
    axes[2, 1].plot(odo[tcol], speed, linewidth=0.8)
    axes[2, 1].set_title("Odometry speed (labeled)")
    axes[2, 1].set_xlabel("time [s]")
    axes[2, 1].set_ylabel("speed [m/s]")
    axes[2, 1].grid(True, alpha=0.4)

    # Create legend for labels using the dynamically assigned colors
    if all_labels:
        legend_patches = [
            mpatches.Patch(color=label_colors[label], label=label, alpha=0.3)
            for label in sorted(all_labels)
        ]
        fig.legend(handles=legend_patches, loc='upper center', 
                  ncol=min(len(legend_patches), 6), bbox_to_anchor=(0.5, 0.98),
                  title='Activity Labels')
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for legend
    
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
    """Save a labeled sensor plot to disk.
    
    Args:
        fig: matplotlib Figure to save
        run_id: Run identifier (used as subdirectory name)
        output_root: Root directory for outputs (e.g., reports/labeled/)
        filename: Output filename (default: 'labeled_sensors_plot.png')
        dpi: Resolution for saved figure (default: 150)
        
    Returns:
        Path to the saved file
    """
    output_root = Path(output_root)
    run_output_dir = output_root / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = run_output_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    return output_path