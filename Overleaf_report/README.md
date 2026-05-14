# Intrinsic Sensor Analysis for Terrain Classification

**Bachelor's Thesis — Technical University of Denmark (DTU)**

A machine learning pipeline for classifying terrain types from IMU (accelerometer + gyroscope) and odometry data collected by a wheeled robot. The pipeline covers the full workflow: raw sensor QC → synchronisation → labelling → cleaning → windowing → feature extraction → model training and hyperparameter tuning.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Data Format](#data-format)
- [Pipeline Stages](#pipeline-stages)
- [Models & Evaluation](#models--evaluation)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)

---

## Project Overview

The goal is to classify terrain types a wheeled robot is driving over using only *intrinsic* sensor signals — i.e., sensors already on the robot (IMU and wheel encoders), without any external camera or LiDAR input.

**Terrain classes:**
- `dry_dirt_track`
- `grass`
- `muddy_dirt_track`
- `smooth_terrain`
- `cobblestone`

**Sensor modalities:**
- Accelerometer (`acc`) — 3-axis, 100 Hz
- Gyroscope (`gyro`) — 3-axis, 100 Hz
- Odometry / wheel encoders (`odo`) — velocity and distance
- Pose (`pose`) — optional position estimates

**Key design decisions:**
- 2-second sliding windows with 50% overlap
- 100+ time-domain and frequency-domain features per window
- Leak-safe cross-validation via `GroupKFold` (groups by robot run)
- Both balanced and unweighted class-weight variants evaluated

---

## Repository Structure

```
.
├── notebooks/                          # Sequential Jupyter pipeline
│   ├── 01_qc.ipynb                     # Raw sensor quality control
│   ├── 02_sync_resample_merge.ipynb    # Synchronise & resample to 100 Hz
│   ├── 03_labelling.ipynb              # Assign terrain labels from time ranges
│   ├── 04_cleaning.ipynb               # Remove outliers & stationary segments
│   ├── 04b_filtering.ipynb             # Optional bandpass / Butterworth filtering
│   ├── 05_windowing.ipynb              # Create sliding windows
│   ├── 06_features.ipynb               # Feature extraction & mutual-information EDA
│   ├── 07_baslines.ipynb               # Baseline model training (LR, RF, XGBoost)
│   ├── 07b_dataset_B_3class_baseline.ipynb  # 3-class dataset variant
│   ├── 08_base_tuning.ipynb            # Hyperparameter tuning
│   └── 09_mlp.ipynb                    # MLP neural network (in progress)
│
├── src/                                # Reusable Python modules
│   ├── features.py                     # Feature engineering (100+ features)
│   ├── io.py                           # Sensor log parsing & I/O
│   ├── labels.py                       # Terrain label assignment
│   ├── preprocess.py                   # QC, synchronisation, resampling
│   └── windowing.py                    # Signal windowing & gap detection
│
├── results/                            # Model metrics & tuning outputs (CSV / JSON)
├── reports/                            # Per-run QC reports
│   ├── raw/                            # QC before preprocessing
│   └── labeled/                        # QC after labelling
│
├── requirements.txt                    # Pip dependencies
├── environment.yml                     # Conda environment
└── README.md
```

---

## Data Format

Each robot run is stored in a directory containing four sensor log files:

| File | Columns | Rate |
|------|---------|------|
| `acc.txt` | `timestamp ax ay az` | ~100 Hz |
| `gyro.txt` | `timestamp gx gy gz` | ~100 Hz |
| `odo.txt` | `timestamp v_left v_right` (or similar) | ~100 Hz |
| `pose.txt` | `timestamp x y theta` | variable |

Terrain labels are defined in a JSON config per run:

```json
{
  "grass":            [[12.5, 45.0], [120.3, 180.1]],
  "dry_dirt_track":   [[50.0, 118.0]],
  "smooth_terrain":   [[185.0, 220.0]]
}
```

Each pair is a `[t_start, t_end]` range in seconds relative to the run start.

---

## Pipeline Stages

Run notebooks in order:

### 1. Quality Control (`01_qc.ipynb`)
- Checks sample rates, timing gaps, jitter, saturation, and DC bias for each sensor
- Outputs per-run QC reports to `reports/raw/`

### 2. Synchronise & Resample (`02_sync_resample_merge.ipynb`)
- Resamples all sensors to a common 100 Hz grid
- Merges accelerometer, gyroscope, and odometry into a single DataFrame per run

### 3. Labelling (`03_labelling.ipynb`)
- Loads JSON label configs and assigns a terrain class to each timestep
- Unlabelled rows (gaps between label ranges) are dropped

### 4. Cleaning (`04_cleaning.ipynb` / `04b_filtering.ipynb`)
- Removes stationary segments (robot not moving)
- Optional Butterworth / bandpass filtering of raw signals

### 5. Windowing (`05_windowing.ipynb`)
- Detects gap-free continuous segments
- Slices each segment into 2-second windows (200 samples at 100 Hz) with 50% overlap
- Assigns the majority terrain label to each window

### 6. Feature Extraction (`06_features.ipynb`)
- Extracts 100+ features per window across all sensor axes and derived magnitudes
- Feature groups: time-domain statistics, FFT spectral features, band power, Hjorth parameters, jerk RMS, cross-sensor correlations
- Mutual information ranking, correlation-based pruning, log1p normalisation for skewed features
- Produces EDA plots (PCA, t-SNE, feature importance heatmaps)

### 7. Baseline Models (`07_baslines.ipynb`)
- Trains Logistic Regression, Random Forest, and XGBoost
- 5-fold `GroupKFold` CV; results saved to `results/`

### 8. Hyperparameter Tuning (`08_base_tuning.ipynb`)
- `RandomizedSearchCV` over all three model families
- Best parameters saved to `results/tuning_best_params.json`

### 9. MLP (`09_mlp.ipynb`) — *in progress*
- Neural network baseline using the same feature set

---

## Models & Evaluation

| Model | Pipeline |
|-------|---------|
| Logistic Regression | `StandardScaler → LogisticRegression` |
| Random Forest | `StandardScaler → RandomForestClassifier` |
| XGBoost | `StandardScaler → XGBClassifier` |

**Cross-validation:** 5-fold `GroupKFold` (grouped by `run_id`) to prevent leakage between runs.

**Metrics reported:** accuracy, macro-F1, per-class precision/recall, and a *robustness score* (mean − std across folds).

**Feature set variants evaluated:**
- All features
- Top-N by mutual information (~25 features)
- Odometry excluded

---

## Setup & Installation

### Option A — pip

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B — Conda

```bash
conda env create -f environment.yml
conda activate terrain-cls
```

### Register the Jupyter kernel (optional)

```bash
python -m ipykernel install --user --name terrain-cls --display-name "Terrain CLS"
```

---

## Usage

1. Place raw run directories under a data root (e.g. `data/runs/run_01/`, `data/runs/run_02/`, …).
2. Create a label JSON for each run following the format described in [Data Format](#data-format).
3. Run notebooks `01` through `08` in order.
4. Results (metrics CSVs, best-params JSON) are written to `results/`.

You can also import the `src` modules directly:

```python
from src.io import read_log, RunData
from src.preprocess import qc_report, qc_stats
from src.labels import load_label_config, label_dataframe
from src.windowing import detect_continuous_segments
from src.features import compute_window_features
```

---

## Results

Metrics from cross-validation runs are stored as CSV files in `results/`:

| File | Description |
|------|-------------|
| `baseline_metrics_*.csv` | Per-fold accuracy & F1 for baseline models |
| `tuned_metrics_*.csv` | Metrics after hyperparameter tuning |
| `dataset_B_3class_*.csv` | 3-class dataset variant results |
| `fold_local_top_mi_features.csv` | Top features ranked by mutual information per fold |
| `fold_quality_diagnostics.csv` | Data quality statistics per fold |
| `tuning_best_params.json` | Best hyperparameters found by RandomizedSearchCV |
