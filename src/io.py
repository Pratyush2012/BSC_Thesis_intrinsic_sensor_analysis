from __future__ import annotations

"""I/O utilities for loading raw robot logs into a consistent in-memory schema.

This module centralizes parsing rules for accelerometer/gyroscope/odometry/pose
text logs so every notebook and script uses the same assumptions.
"""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


ACC_FILE = "log_t0_acc_1.txt"
GYRO_FILE = "log_t0_gyro_1.txt"
ODO_FILE = "log_t0_encoder_velocity.txt"
POSE_FILE = "log_t0_pose.txt"


@dataclass(frozen=True)
class RunData:
    """Container for one run with harmonized sensor DataFrames.

    The `t_rel` column in each frame is aligned to a shared run-level start time.
    """

    run_id: str
    run_dir: Path
    acc: pd.DataFrame
    gyro: pd.DataFrame
    odo: pd.DataFrame
    pose: pd.DataFrame | None = None

    def sensors(self) -> dict[str, pd.DataFrame]:
        """Return available sensors in a simple dict for iteration/reporting."""

        sensors = {"acc": self.acc, "gyro": self.gyro, "odo": self.odo}
        if self.pose is not None:
            sensors["pose"] = self.pose
        return sensors


def read_log(path: str | Path, cols: list[str], usecols: list[int] | None = None) -> pd.DataFrame:
    """Parse one raw whitespace-separated log file.

    Notes:
    - `%` prefixed rows are metadata/comments and are ignored.
    - We sort and deduplicate by timestamp to enforce monotonic samples.
    - Empty parsed files raise early so callers can skip bad runs deterministically.
    """

    path = Path(path)
    df = pd.read_csv(
        path,
        comment="%",
        sep=r"\s+",
        header=None,
        names=cols,
        usecols=usecols,
        engine="python",
    )
    if df.empty:
        raise ValueError(f"No data rows found in {path}")

    time_col = cols[0]
    # Keep the first sample for repeated timestamps to maintain deterministic behavior.
    df = (
        df.dropna()
        .sort_values(time_col)
        .drop_duplicates(time_col, keep="first")
        .reset_index(drop=True)
    )
    return df


def discover_run_dirs(raw_root: str | Path) -> list[Path]:
    """Discover run folders under the raw root.

    We treat folders named like `log_YYYYMMDD_HHMMSS.xxx` as run containers.
    """

    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {raw_root}")

    # `rglob("log_*")` finds runs regardless of the dataset subfolder layout.
    run_dirs = [p for p in raw_root.rglob("log_*") if p.is_dir()]
    run_dirs = sorted({p.resolve() for p in run_dirs})
    return run_dirs


def _add_relative_time(df: pd.DataFrame, t0: float, tcol: str = "t") -> pd.DataFrame:
    """Add a run-relative time column so all sensors share a common timeline."""

    out = df.copy()
    out["t_rel"] = out[tcol] - t0
    return out


def load_run(run_dir: str | Path, include_pose: bool = True) -> RunData:
    """Load one run directory into `RunData`.

    Required logs:
    - accelerometer
    - gyroscope
    - encoder velocity

    Optional:
    - pose (loaded when present and `include_pose=True`)
    """

    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    acc_path = run_dir / ACC_FILE
    gyro_path = run_dir / GYRO_FILE
    odo_path = run_dir / ODO_FILE
    pose_path = run_dir / POSE_FILE

    required = [acc_path, gyro_path, odo_path]
    missing_required = [str(p) for p in required if not p.exists()]
    if missing_required:
        missing_str = ", ".join(missing_required)
        raise FileNotFoundError(f"Missing required log files: {missing_str}")

    # Column names are the canonical schema used across the pipeline.
    acc = read_log(acc_path, ["t", "ax", "ay", "az"])
    gyro = read_log(gyro_path, ["t", "gx", "gy", "gz"])
    # Odo file has debug columns after v1/v2; we intentionally keep only 0/1/2.
    odo = read_log(odo_path, ["t", "v1", "v2"], usecols=[0, 1, 2])

    pose: pd.DataFrame | None = None
    if include_pose and pose_path.exists():
        pose = read_log(pose_path, ["t", "x", "y", "heading", "tilt"])

    first_times = [acc["t"].iloc[0], gyro["t"].iloc[0], odo["t"].iloc[0]]
    if pose is not None:
        first_times.append(pose["t"].iloc[0])
    t0 = float(min(first_times))

    # Align each sensor to a shared `t_rel` origin at earliest sensor start.
    acc = _add_relative_time(acc, t0=t0, tcol="t")
    gyro = _add_relative_time(gyro, t0=t0, tcol="t")
    odo = _add_relative_time(odo, t0=t0, tcol="t")
    if pose is not None:
        pose = _add_relative_time(pose, t0=t0, tcol="t")

    return RunData(
        run_id=run_dir.name,
        run_dir=run_dir,
        acc=acc,
        gyro=gyro,
        odo=odo,
        pose=pose,
    )


def load_runs(raw_root: str | Path, include_pose: bool = True) -> list[RunData]:
    """Load all valid runs from a raw root.

    Invalid runs (missing files, empty logs) are skipped rather than failing
    the full batch load.
    """

    runs: list[RunData] = []
    for run_dir in discover_run_dirs(raw_root):
        try:
            runs.append(load_run(run_dir, include_pose=include_pose))
        # Bad runs are expected in exploratory datasets; skip and continue.
        except (FileNotFoundError, ValueError):
            continue
    return runs


def save_labeled_run(run_data: RunData, output_root: str | Path) -> None:
    """Save a RunData object with labeled sensor data to CSV files.

    Mirroring the raw directory structure, creates:
    - {output_root}/{run_data.run_id}/log_t0_acc_1.csv
    - {output_root}/{run_data.run_id}/log_t0_gyro_1.csv
    - {output_root}/{run_data.run_id}/log_t0_encoder_velocity.csv
    - {output_root}/{run_data.run_id}/log_t0_pose.csv (if pose exists)

    Args:
        run_data: RunData object with labeled sensor DataFrames
        output_root: Root directory where labeled data will be saved
    """
    output_root = Path(output_root)
    run_output_dir = output_root / run_data.run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Save each sensor to CSV
    run_data.acc.to_csv(run_output_dir / "log_t0_acc_1.csv", index=False)
    run_data.gyro.to_csv(run_output_dir / "log_t0_gyro_1.csv", index=False)
    run_data.odo.to_csv(run_output_dir / "log_t0_encoder_velocity.csv", index=False)
    
    if run_data.pose is not None:
        run_data.pose.to_csv(run_output_dir / "log_t0_pose.csv", index=False)
