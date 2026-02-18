"""Label assignment utilities for sensor data."""

from dataclasses import dataclass
from typing import List, Optional, Dict
import json
import pandas as pd


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
        Label name if timestamp falls in a range, None otherwise
    """
    for label_name, config in label_configs.items():
        for label_range in config.ranges:
            if label_range.start <= timestamp <= label_range.end:
                return label_name
    return None


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
