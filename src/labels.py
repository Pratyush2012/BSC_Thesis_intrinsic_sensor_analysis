"""Label assignment utilities for sensor data."""

from dataclasses import dataclass
from typing import List, Optional, Dict, TYPE_CHECKING
from pathlib import Path
import json
import pandas as pd


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
        'transition_samples': transition_samples    }