# Intrinsic Sensor Analysis for Terrain Classification

**Bachelor's Thesis — Technical University of Denmark (DTU)**

A machine learning pipeline for classifying terrain types from IMU (accelerometer + gyroscope) and odometry data collected by a wheeled robot. The pipeline covers the full workflow: raw sensor QC → synchronisation → labelling → cleaning → windowing → feature extraction → classical baselines & tuning → neural networks (MLP, CNN) → model comparison → online transition detection.

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

**Terrain classes (5-class problem):**
- `dry_dirt_track`
- `grass`
- `muddy_dirt_track`
- `smooth_terrain`
- `cobblestone`

A reduced **3-class variant** (Dataset B) merges fine-grained subclasses and restricts to low-speed, near-constant-velocity windows to test robustness when the speed confound is removed.

**Sensor modalities:**
- Accelerometer (`acc`) — 3-axis, ~100 Hz
- Gyroscope (`gyro`) — 3-axis, ~100 Hz
- Odometry / wheel encoders (`odo`) — left/right wheel velocity
- Pose (`pose`) — optional position estimates

**Key design decisions:**
- 2-second sliding windows with 50% overlap (200 samples at 100 Hz)
- 130+ time-domain and frequency-domain features per window
- Leak-safe cross-validation via Leave-One-Run-Out (LORO) grouped by `run_id`
- Both balanced and unweighted class-weight variants evaluated
- Velocity-regime split into Dataset A (all windows) and Dataset B (low, near-constant speed)
- Online CUSUM transition detector (Blanke et al. 2016, §6.4) for change-point analysis

---

## Repository Structure

```
.
├── notebooks/                                      # Sequential Jupyter pipeline
│   ├── 01_qc.ipynb                                 # Raw sensor quality control
│   ├── 02_sync_resample_merge.ipynb                # Synchronise & resample to 100 Hz
│   ├── 03_labelling.ipynb                          # Assign terrain labels from time ranges
│   ├── 04_cleaning.ipynb                           # Remove outliers & stationary segments
│   ├── 04b_filtering.ipynb                         # Optional bandpass / Butterworth filtering
│   ├── 05_windowing.ipynb                          # Create sliding windows
│   ├── 06_features.ipynb                           # Feature extraction & MI/EDA (5-class)
│   ├── 06b_features_dataset_B_3class.ipynb         # Feature extraction for Dataset B
│   ├── 07_baslines.ipynb                           # Baseline models (LR, RF, XGBoost, SVM)
│   ├── 07a_dataset_A_3class_baseline.ipynb         # Dataset A 3-class baseline
│   ├── 07b_dataset_B_3class_baseline.ipynb         # Dataset B 3-class baseline
│   ├── 08_base_tuning.ipynb                        # RandomizedSearchCV hyperparameter tuning
│   ├── 08b_base_tuning_dataset_B_3class.ipynb      # Tuning for Dataset B
│   ├── 09_mlp.ipynb                                # MLP neural network
│   ├── 09b_mlp_dataset_B_3class.ipynb              # MLP for Dataset B
│   ├── 10_CNN.ipynb                                # 1D CNN from raw windows
│   ├── 10b_CNN_dataset_B_3class.ipynb              # CNN for Dataset B
│   ├── 11_model_comparison.ipynb                   # Cross-model comparison & figures
│   └── 12_transition_analysis.ipynb                # Online CUSUM transition detection
│
├── src/                                            # Reusable Python modules
│   ├── io.py                                       # Sensor log parsing & I/O
│   ├── preprocess.py                               # QC, synchronisation, resampling
│   ├── labels.py                                   # Terrain label assignment & plots
│   ├── windowing.py                                # Signal windowing & gap detection
│   ├── features.py                                 # Feature engineering (130+ features)
│   ├── evaluation.py                               # Shared model factories & metrics
│   ├── transition.py                               # LORO-CV prediction + transition analysis
│   └── change_detection.py                         # Online CUSUM bank-of-detectors
│
├── results/                                        # Model metrics & posteriors (CSV / NPZ)
│   ├── Farm/                                       # Farm dataset (5-class)
│   ├── Campus/                                     # Campus dataset (5-class)
│   ├── Campus_dataset_B_3class/                    # 3-class low-speed variant
│   └── comparison/                                 # Cross-dataset aggregates
│
├── reports/                                        # Generated figures (PNG)
│   ├── raw/                                        # Per-run QC sanity plots
│   ├── features/                                   # MI, PCA/t-SNE, distribution figures
│   └── models/                                     # Confusion matrices, learning curves, saliency
│
├── requirements.txt                                # Pip dependencies
├── environment.yml                                 # Conda environment
└── README.md
```

> **Not included in the repository** (kept local; regenerated by running the pipeline):
> - `data/` — raw, interim, and processed datasets (sensor logs are private)
> - `models/` — saved sklearn / PyTorch model artefacts
> - `reports/labeled/` and `reports/raw/Test/` — large per-run plots for exploratory runs

---

## Data Format

Each robot run is stored in a directory containing four sensor log files:

| File | Columns | Rate |
|------|---------|------|
| `log_t0_acc_1.txt` | `timestamp ax ay az` | ~100 Hz |
| `log_t0_gyro_1.txt` | `timestamp gx gy gz` | ~100 Hz |
| `log_t0_encoder_velocity.txt` | `timestamp v1 v2 ...` | ~100 Hz |
| `log_t0_pose.txt` | `timestamp x y heading tilt` | variable |

Terrain labels are defined in a JSON config per run (`labels_config.json`):

```json
{
  "grass":            [[12.5, 45.0], [120.3, 180.1]],
  "dry_dirt_track":   [[50.0, 118.0]],
  "smooth_terrain":   [[185.0, 220.0]]
}
```

Each pair is a `[t_start, t_end]` range in seconds relative to the run start.

**Physical axis convention (hardcoded in `src/features.py`):**
- Accelerometer: `ax` → lateral (left/right), `ay` → vertical (gravity axis), `az` → longitudinal (front/back)
- Gyroscope: `gx` → pitch, `gy` → yaw, `gz` → roll
- `gy` (yaw rate) is excluded from features — it encodes turning, not terrain.

---

## Pipeline Stages

Run notebooks in order:

### 1. Quality Control (`01_qc.ipynb`)
- Checks sample rates, timing gaps, jitter, saturation, and DC bias for each sensor
- Outputs per-run QC reports to `reports/raw/`
- Builds a dataset-level pass/fail table; failing runs are excluded downstream

### 2. Synchronise & Resample (`02_sync_resample_merge.ipynb`)
- Resamples all sensors to a common 100 Hz grid (linear interpolation)
- Merges accelerometer, gyroscope, and odometry into a single DataFrame per run

### 3. Labelling (`03_labelling.ipynb`)
- Loads JSON label configs and assigns a terrain class to each timestep
- Generates per-run labelled-sensor visualisations (local-only)

### 4. Cleaning (`04_cleaning.ipynb` / `04b_filtering.ipynb`)
- Removes stationary, low-speed, and high-yaw-rate segments
- Optional Butterworth / bandpass filtering of raw signals

### 5. Windowing (`05_windowing.ipynb`)
- Detects gap-free continuous segments
- Slices each segment into 2-second windows (200 samples) with 50% overlap
- Assigns the majority terrain label to each window

### 6. Feature Extraction (`06_features.ipynb`, `06b_features_dataset_B_3class.ipynb`)
- Extracts 130+ features per window across all sensor axes and derived magnitudes
- Feature groups: time-domain statistics, FFT spectral features, band power, Hjorth parameters, jerk RMS, within- and cross-sensor correlations, odometry summaries
- Mutual information ranking, correlation-based pruning, log1p normalisation for skewed features
- Produces EDA plots (PCA, t-SNE, feature importance, distribution histograms/violins) under `reports/features/`
- Splits into Dataset A (all windows) and Dataset B (low-speed, near-constant-speed subset)

### 7. Baseline Models (`07_baslines.ipynb`, `07a_*`, `07b_*`)
- Trains Logistic Regression, Random Forest, XGBoost, and SVM
- LORO-CV; results saved to `results/`

### 8. Hyperparameter Tuning (`08_base_tuning.ipynb`, `08b_*`)
- `RandomizedSearchCV` with `GroupKFold` over all model families
- Best parameters saved to `results/<dataset>/tuning_best_params.json`

### 9. MLP (`09_mlp.ipynb`, `09b_*`)
- PyTorch MLP on engineered features
- Linear → BatchNorm → ReLU → Dropout stack with class-weighted cross-entropy

### 10. CNN (`10_CNN.ipynb`, `10b_*`)
- PyTorch 1D CNN on raw IMU windows (5 channels × 200 samples)
- Conv1d → BN → ReLU → MaxPool blocks + AdaptiveAvgPool + FC head

### 11. Model Comparison (`11_model_comparison.ipynb`)
- Aggregates metrics across all model families and datasets
- Per-class confusion matrices, robustness scores, noise sensitivity, sensor-dropout ablations, cross-day performance

### 12. Transition Analysis (`12_transition_analysis.ipynb`)
- Stable-detection delay analysis (n_stable consecutive correct predictions)
- Online CUSUM bank-of-detectors (Blanke et al. 2016, §6.4) calibrated to a target false-alarm rate
- Operating-curve sweep over detector threshold `h`

---

## Models & Evaluation

| Model | Pipeline |
|-------|---------|
| Logistic Regression | `StandardScaler → LogisticRegression` |
| Random Forest | `StandardScaler → RandomForestClassifier` |
| XGBoost | `StandardScaler → XGBClassifier` |
| SVM | `StandardScaler → SVC (RBF kernel)` |
| MLP | `StandardScaler → TerrainMLP` (PyTorch) |
| 1D CNN | raw signals → `TerrainCNN1D` (PyTorch) |

**Cross-validation:** Leave-One-Run-Out (LORO) — grouped by `run_id` to prevent leakage between runs.

**Metrics reported:** accuracy, macro-F1, per-class precision/recall/F1, and a *robustness score* (`macro_f1_mean − 0.5 × macro_f1_std`).

**Feature set variants evaluated:**
- All features
- Top-N by mutual information
- Odometry excluded
- 5-class vs 3-class label sets
- Dataset A (all speeds) vs Dataset B (controlled low speed)

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

The raw sensor logs are not included in this repository. To reproduce the pipeline end-to-end you need to provide your own runs under `data/raw/<dataset>/<run_name>/` following the [Data Format](#data-format) above, then:

1. Place raw run directories under `data/raw/` (e.g. `data/raw/Farm/Run1/log_*/`).
2. Create a `labels_config.json` for each run.
3. Run notebooks `01` through `12` in order.
4. Results (metrics CSVs, posteriors, best-params JSON) are written to `results/`; figures to `reports/`.

If you only want to inspect the produced metrics, the CSV/NPZ files under `results/` and the figures under `reports/` are sufficient on their own — you do not need to re-run the pipeline.

You can also import the `src` modules directly:

```python
from src.io import load_run, load_runs, save_labeled_run
from src.preprocess import qc_report, resample_sensors, build_dataset_qc_table
from src.labels import load_label_config, label_dataframe, discover_labeled_runs
from src.windowing import detect_continuous_segments, create_windows
from src.features import compute_window_features, run_feature_evaluation
from src.evaluation import make_base_models, compute_fold_metrics
from src.transition import analyze_transitions, compare_models
from src.change_detection import cusum_bank, calibrate_threshold
```

---

## Results

Metrics, posteriors, and figures are organised under `results/` and `reports/` by dataset:

- `Farm/` — outdoor farm dataset (5 classes)
- `Campus/` — campus dataset (5 classes)
- `Campus_dataset_B_3class/` — 3-class low-speed variant
- `comparison/` — cross-dataset aggregates

Per-dataset result files include:

| File | Description |
|------|-------------|
| `baseline_metrics_per_fold.csv` / `_summary.csv` | Per-fold and aggregated accuracy & F1 for baseline models |
| `tuned_metrics_*.csv` | Metrics after hyperparameter tuning |
| `mlp_metrics_*.csv` | MLP per-fold and summary metrics |
| `cnn/cnn_metrics_*.csv` | CNN per-fold and summary metrics |
| `per_class_metrics_<model>.csv` | Per-class precision/recall/F1 per model |
| `fold_local_top_mi_features.csv` | Top features ranked by mutual information per fold |
| `fold_quality_diagnostics.csv` | Data quality statistics per fold |
| `proba_<model>.npz` | Soft posteriors used by the CUSUM detector in notebook 12 |
| `master_model_comparison.csv` | Side-by-side model comparison within the dataset |
| `noise_sensitivity.csv`, `sensor_dropout.csv`, `cross_day_performance.csv` | Robustness ablations |
| `transition_analysis_summary.csv` | Stable-detection and CUSUM transition outcomes |
| `tuning_best_params.json` | Best hyperparameters found by RandomizedSearchCV |

Figures (confusion matrices, learning curves, MI heatmaps, PCA/t-SNE, CNN saliency, etc.) are saved under `reports/features/` and `reports/models/`.
