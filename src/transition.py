"""
src/transition.py
-----------------
Terrain transition detection analysis for windowed IMU + odometry data.

A 'ground-truth transition' is defined as any consecutive pair of windows
(sorted by t_start within a run) where the true label changes.

A 'stable detection' requires n_stable consecutive predicted windows that all
carry the target label, starting from the first window of the new GT epoch.
This filters noisy single-window misclassifications.

All transition analysis functions operate on per-run DataFrames. The notebook
is responsible for the LORO-CV loop that produces pred_label for each window.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------------------------
# Optional deep-learning imports (only needed for MLP / CNN prediction generation)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.evaluation import make_base_models

# ===========================================================================
# CONSTANTS
# ===========================================================================

LABEL_ORDER: list[str] = [
    "cobblestone",
    "dry_dirt_track",
    "grass",
    "muddy_dirt_track",
    "smooth_terrain",
]

TERRAIN_COLORS: dict[str, str] = {
    "cobblestone":      "#e07b39",
    "dry_dirt_track":   "#c4a35a",
    "grass":            "#5aab61",
    "muddy_dirt_track": "#7b5ea7",
    "smooth_terrain":   "#4f8bc9",
}

ID_COLS: list[str] = [
    "window_id", "run_id", "segment_id", "label",
    "t_start", "t_end", "n_samples",
]

# Signal channels used by the CNN (gy/yaw excluded — encodes steering, not terrain)
SIGNAL_COLS: list[str] = ["ax", "ay", "az", "gx", "gz"]

# ===========================================================================
# PYTORCH MODEL CLASSES
# (Copied verbatim from notebooks 09_mlp.ipynb and 10_CNN.ipynb)
# ===========================================================================

if _TORCH_AVAILABLE:

    class TerrainDataset(Dataset):
        """PyTorch Dataset wrapper for terrain feature data (MLP input)."""

        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx: int):
            return self.X[idx], self.y[idx]

    class TerrainMLP(nn.Module):
        """
        Multi-Layer Perceptron for terrain classification.

        Architecture: Linear → BatchNorm → ReLU → Dropout → ... → Linear (raw logits).
        CrossEntropyLoss applies softmax internally; the output layer has no activation.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer widths, e.g. [256, 128, 64].
            num_classes: Number of terrain classes.
            dropout_rate: Dropout probability.
            use_batch_norm: Apply BatchNorm after each linear layer.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int],
            num_classes: int,
            dropout_rate: float = 0.3,
            use_batch_norm: bool = True,
        ):
            super().__init__()
            layers: list[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_rate))
                prev = h
            self.hidden_layers = nn.Sequential(*layers)
            self.output_layer = nn.Linear(prev, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.output_layer(self.hidden_layers(x))

    class TerrainCNN1D(nn.Module):
        """
        1D Convolutional Neural Network for terrain classification from raw IMU windows.

        Architecture:
            Variable convolutional blocks (Conv1d → BN → ReLU → MaxPool),
            AdaptiveAvgPool, then a 2-layer FC head (raw logits).

        Args:
            in_channels: Number of input sensor channels (default 5: ax,ay,az,gx,gz).
            num_classes: Number of terrain classes.
            filters: Output channels per conv block (default [32, 64, 128]).
            kernels: Kernel size per conv block (default [7, 5, 3]).
            dropout_rate: Dropout probability in the FC head.
        """

        def __init__(
            self,
            in_channels: int = 5,
            num_classes: int = 5,
            filters: list | None = None,
            kernels: list | None = None,
            dropout_rate: float = 0.3,
        ):
            super().__init__()
            if filters is None:
                filters = [32, 64, 128]
            if kernels is None:
                kernels = [7, 5, 3]
            assert len(filters) == len(kernels)

            blocks = []
            ch_in = in_channels
            for ch_out, k in zip(filters, kernels):
                blocks.append(nn.Sequential(
                    nn.Conv1d(ch_in, ch_out, kernel_size=k, padding=k // 2, bias=False),
                    nn.BatchNorm1d(ch_out),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                ))
                ch_in = ch_out
            self.blocks = nn.ModuleList(blocks)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.Linear(filters[-1], 64),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(64, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = block(x)
            x = self.global_pool(x).squeeze(-1)
            return self.head(x)

    # -----------------------------------------------------------------------
    # Training utilities
    # -----------------------------------------------------------------------

    def _train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
        """Train for one epoch. Returns mean batch loss."""
        model.train()
        total_loss, batch_count = 0.0, 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss = criterion(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        return total_loss / max(batch_count, 1)

    def _evaluate(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[float, np.ndarray]:
        """Evaluate model. Returns (mean_loss, predicted_class_indices)."""
        model.eval()
        total_loss, batch_count = 0.0, 0
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                total_loss += criterion(logits, y_b).item()
                batch_count += 1
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return total_loss / max(batch_count, 1), np.concatenate(preds)

    def _build_X_tensor(df: pd.DataFrame) -> torch.Tensor:
        """
        Stack raw signal series columns into a float32 tensor of shape (N, C, T).

        Expects df to have columns '{col}__series' for each col in SIGNAL_COLS,
        where each cell is a 1-D numpy array of length T (= 200 for 2-second windows).
        """
        arrays = [np.stack(df[f"{c}__series"].values) for c in SIGNAL_COLS]
        return torch.tensor(np.stack(arrays, axis=1), dtype=torch.float32)

    def _train_with_early_stopping(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        max_epochs: int,
        patience: int,
    ) -> nn.Module:
        """
        Train model with early stopping. Restores best-validation-loss weights.
        Returns the trained model.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict | None = None

        for _ in range(max_epochs):
            _train_one_epoch(model, train_loader, optimizer, criterion, device)
            v_loss, _ = _evaluate(model, val_loader, criterion, device)
            if scheduler is not None:
                scheduler.step()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model


# ===========================================================================
# PREDICTION GENERATION — CLASSICAL MODELS (sklearn Pipelines)
# ===========================================================================

def run_classical_loro_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    run_col: str = "run_id",
    model_name: str = "RandomForest",
    imbalance: str = "balanced",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate per-window predictions for a classical model via LORO-CV.

    Mirrors the evaluation loop in 07_baslines.ipynb exactly:
    - StandardScaler + classifier in a Pipeline (fitted on train fold only).
    - XGBoost: LabelEncoder + sample weights for balanced mode.
    - Others: class_weight='balanced' when imbalance='balanced'.

    Args:
        df: Feature DataFrame (dataset_A_pruned_5run.csv).
        feature_cols: Feature column names.
        label_col: Ground-truth label column.
        run_col: Run identifier column.
        model_name: One of 'RandomForest', 'XGBoost', 'LogisticRegression', 'SVM'.
        imbalance: 'balanced' or 'unweighted'.
        random_state: Reproducibility seed.

    Returns:
        Copy of df with added 'pred_label' column (string labels, index-aligned).
    """
    run_ids = sorted(df[run_col].unique().tolist())
    pred_labels = pd.Series(index=df.index, dtype=object)
    is_xgb = model_name == "XGBoost"

    for test_run in run_ids:
        train_mask = df[run_col].astype(str) != str(test_run)
        test_mask = ~train_mask

        X_train = df.loc[train_mask, feature_cols]
        X_test = df.loc[test_mask, feature_cols]
        y_train_str = df.loc[train_mask, label_col].astype(str).to_numpy()
        y_test_str = df.loc[test_mask, label_col].astype(str).to_numpy()

        models = make_base_models(random_state)
        pipeline = models[model_name]

        if is_xgb:
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train_str)
            if imbalance == "balanced":
                sw = compute_sample_weight("balanced", y_train_enc)
                pipeline.fit(X_train, y_train_enc, clf__sample_weight=sw)
            else:
                pipeline.fit(X_train, y_train_enc)
            y_pred_enc = pipeline.predict(X_test)
            # Clip to valid range in case of any encoding edge case
            y_pred_enc = np.clip(y_pred_enc, 0, len(le.classes_) - 1).astype(int)
            y_pred = le.inverse_transform(y_pred_enc)
        else:
            if imbalance == "balanced":
                try:
                    pipeline.set_params(clf__class_weight="balanced")
                except ValueError:
                    pass  # SVC with random_state doesn't always expose this cleanly
            pipeline.fit(X_train, y_train_str)
            y_pred = pipeline.predict(X_test)

        pred_labels.loc[test_mask] = y_pred

    result = df.copy()
    result["pred_label"] = pred_labels.values
    return result


# ===========================================================================
# PREDICTION GENERATION — MLP
# ===========================================================================

def run_mlp_loro_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    run_col: str = "run_id",
    hidden_dims: list[int] | None = None,
    dropout_rate: float = 0.3,
    batch_size: int = 32,
    max_epochs: int = 200,
    patience: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate per-window predictions for the MLP via LORO-CV.

    Mirrors the training loop in 09_mlp.ipynb exactly:
    - StandardScaler fitted on train fold only.
    - LabelEncoder fitted on train fold only.
    - Class weights: N / (n_classes * count[i]).
    - 85/15 stratified train/val split; early stopping on val loss.
    - CosineAnnealingLR scheduler.

    Args:
        df: Feature DataFrame with metadata + feature columns.
        feature_cols: List of feature column names.
        label_col: Ground-truth label column.
        run_col: Run identifier column.
        hidden_dims: Hidden layer widths (default [256, 128, 64]).
        dropout_rate: Dropout probability (default 0.3).
        batch_size: Mini-batch size (default 32).
        max_epochs: Maximum training epochs (default 200).
        patience: Early stopping patience (default 20).
        learning_rate: Adam learning rate (default 1e-3).
        weight_decay: L2 regularisation (default 1e-4).
        val_split: Validation fraction within training fold (default 0.15).
        random_state: Reproducibility seed.

    Returns:
        Copy of df with added 'pred_label' column.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for run_mlp_loro_cv. Install torch.")

    if hidden_dims is None:
        hidden_dims = [256, 128, 64]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_ids = sorted(df[run_col].unique().tolist())
    pred_labels = pd.Series(index=df.index, dtype=object)

    for test_run in run_ids:
        train_mask = df[run_col].astype(str) != str(test_run)
        test_mask = ~train_mask

        y_train_str = df.loc[train_mask, label_col].astype(str).to_numpy()
        y_test_str = df.loc[test_mask, label_col].astype(str).to_numpy()

        # Scale features (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(df.loc[train_mask, feature_cols])
        X_test_scaled = scaler.transform(df.loc[test_mask, feature_cols])

        # Stratified train/val split
        X_tr, X_val, y_tr_str, y_val_str = train_test_split(
            X_train_scaled, y_train_str,
            test_size=val_split,
            stratify=y_train_str,
            random_state=random_state,
        )

        # Label encoding (fit on train split only)
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_tr_str)
        y_val_enc = le.transform(y_val_str)
        y_test_enc = le.transform(y_test_str)

        # Class weights
        n_classes = len(le.classes_)
        counts = np.bincount(y_tr_enc, minlength=n_classes).astype(float)
        cw = len(y_tr_enc) / (n_classes * np.maximum(counts, 1))
        cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

        # Data loaders
        train_loader = DataLoader(TerrainDataset(X_tr, y_tr_enc), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TerrainDataset(X_val, y_val_enc), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TerrainDataset(X_test_scaled, y_test_enc), batch_size=batch_size, shuffle=False)

        # Model
        model = TerrainMLP(
            input_dim=X_tr.shape[1],
            hidden_dims=hidden_dims,
            num_classes=n_classes,
            dropout_rate=dropout_rate,
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=cw_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=learning_rate * 0.01
        )

        model = _train_with_early_stopping(
            model, train_loader, val_loader, criterion, optimizer,
            scheduler, device, max_epochs, patience,
        )

        # Inference
        _, y_pred_enc = _evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
        y_pred = le.inverse_transform(y_pred_enc)
        pred_labels.loc[test_mask] = y_pred

        print(f"  MLP fold {test_run}: acc={np.mean(y_pred == y_test_str):.3f}")

    result = df.copy()
    result["pred_label"] = pred_labels.values
    return result


# ===========================================================================
# PREDICTION GENERATION — CNN
# ===========================================================================

def run_cnn_loro_cv(
    feature_df: pd.DataFrame,
    cnn_df: pd.DataFrame,
    label_col: str = "label",
    run_col: str = "run_id",
    filters: list[int] | None = None,
    kernels: list[int] | None = None,
    dropout_rate: float = 0.3,
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: int = 15,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate per-window predictions for the CNN via LORO-CV.

    Mirrors the training loop in 10_CNN.ipynb exactly:
    - No signal normalisation (CNN learns scale from data).
    - LabelEncoder fitted on train fold only.
    - Class weights: N / (n_classes * count[i]).
    - 85/15 stratified train/val split; early stopping on val loss.
    - CosineAnnealingLR scheduler.

    Args:
        feature_df: Feature DataFrame (dataset_A_pruned_5run.csv) — used for
                    metadata alignment (run_id, label, t_start, index).
        cnn_df: Raw-window parquet DataFrame (raw_windows_cnn.parquet) — must
                contain '{col}__series' columns for each col in SIGNAL_COLS.
                Its index must align with feature_df (same window_id order).
        label_col: Ground-truth label column.
        run_col: Run identifier column.
        filters: Conv filter sizes (default [32, 64, 128]).
        kernels: Conv kernel sizes (default [7, 5, 3]).
        dropout_rate: Dropout probability (default 0.3).
        batch_size: Mini-batch size (default 32).
        max_epochs: Maximum training epochs (default 100).
        patience: Early stopping patience (default 15).
        learning_rate: Adam learning rate (default 1e-3).
        weight_decay: L2 regularisation (default 1e-4).
        val_split: Validation fraction (default 0.15).
        random_state: Reproducibility seed.

    Returns:
        Copy of feature_df with added 'pred_label' column (index-aligned).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for run_cnn_loro_cv. Install torch.")

    if filters is None:
        filters = [32, 64, 128]
    if kernels is None:
        kernels = [7, 5, 3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_ids = sorted(feature_df[run_col].unique().tolist())

    # Align cnn_df to feature_df by window_id
    if "window_id" in feature_df.columns and "window_id" in cnn_df.columns:
        cnn_df = cnn_df.set_index("window_id").loc[feature_df["window_id"].values].reset_index()
        cnn_df.index = feature_df.index
    else:
        # Assume order already matches
        cnn_df = cnn_df.copy()
        cnn_df.index = feature_df.index

    pred_labels = pd.Series(index=feature_df.index, dtype=object)

    for test_run in run_ids:
        train_mask = feature_df[run_col].astype(str) != str(test_run)
        test_mask = ~train_mask

        y_train_str = feature_df.loc[train_mask, label_col].astype(str).to_numpy()
        y_test_str = feature_df.loc[test_mask, label_col].astype(str).to_numpy()

        # Build (N, C, T) signal tensors
        X_train_t = _build_X_tensor(cnn_df.loc[train_mask])
        X_test_t = _build_X_tensor(cnn_df.loc[test_mask])

        # Label encoding
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train_str)
        y_test_enc = le.transform(y_test_str)

        # Stratified train/val split
        X_tr, X_val, y_tr_enc, y_val_enc = train_test_split(
            X_train_t, y_train_enc,
            test_size=val_split,
            stratify=y_train_enc,
            random_state=random_state,
        )

        # Class weights
        n_classes = len(le.classes_)
        counts = np.bincount(y_tr_enc, minlength=n_classes).astype(float)
        cw = len(y_tr_enc) / (n_classes * np.maximum(counts, 1))
        cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

        # Data loaders
        train_ds = TensorDataset(X_tr, torch.from_numpy(y_tr_enc.astype(np.int64)))
        val_ds = TensorDataset(X_val, torch.from_numpy(y_val_enc.astype(np.int64)))
        test_ds = TensorDataset(X_test_t, torch.from_numpy(y_test_enc.astype(np.int64)))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Model
        model = TerrainCNN1D(
            in_channels=len(SIGNAL_COLS),
            num_classes=n_classes,
            filters=filters,
            kernels=kernels,
            dropout_rate=dropout_rate,
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=cw_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=learning_rate * 0.01
        )

        model = _train_with_early_stopping(
            model, train_loader, val_loader, criterion, optimizer,
            scheduler, device, max_epochs, patience,
        )

        # Inference
        _, y_pred_enc = _evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
        y_pred = le.inverse_transform(y_pred_enc)
        pred_labels.loc[test_mask] = y_pred

        print(f"  CNN fold {test_run}: acc={np.mean(y_pred == y_test_str):.3f}")

    result = feature_df.copy()
    result["pred_label"] = pred_labels.values
    return result


# ===========================================================================
# TRANSITION DETECTION FUNCTIONS
# ===========================================================================

def detect_gt_transitions(
    df: pd.DataFrame,
    time_col: str = "time",
    true_col: str = "true_label",
) -> list[dict]:
    """
    Detect ground-truth terrain transitions in a single-run, time-sorted DataFrame.

    A transition is any position i where df[true_col].iloc[i] != df[true_col].iloc[i-1]
    (NaN labels are skipped). The DataFrame must be pre-sorted by time_col ascending.

    Args:
        df: Single-run DataFrame sorted by time_col.
        time_col: Column with window start time (float, seconds).
        true_col: Column with ground-truth label (str).

    Returns:
        List of dicts with keys:
            transition_idx  — iloc of the first window of the new terrain epoch.
            from_label      — label of the epoch that ended.
            to_label        — label of the new epoch.
            gt_time         — df[time_col].iloc[transition_idx].
        Empty list if df has fewer than 2 rows or no label changes.
    """
    if len(df) < 2:
        return []

    transitions: list[dict] = []
    labels = df[true_col].to_numpy()
    times = df[time_col].to_numpy()

    prev_label = None
    for i, lbl in enumerate(labels):
        if pd.isna(lbl):
            continue
        if prev_label is not None and lbl != prev_label:
            transitions.append({
                "transition_idx": i,
                "from_label": prev_label,
                "to_label": str(lbl),
                "gt_time": float(times[i]),
            })
        prev_label = str(lbl)

    return transitions


def find_stable_detection(
    series: pd.Series | np.ndarray,
    start_iloc: int,
    target_label: str,
    n_stable: int = 3,
    end_iloc: int | None = None,
) -> int | None:
    """
    Find the first iloc in series where n_stable consecutive values equal target_label.

    Implements the 'stable detection' criterion: a single stray prediction does
    not count — the model must commit to the new class for n_stable windows in a
    row before the transition is considered detected.

    Args:
        series: Array or Series of predicted label strings.
        start_iloc: Positional index to begin searching from (inclusive).
        target_label: Label that must appear n_stable times consecutively.
        n_stable: Minimum run length required (>= 1).
        end_iloc: Exclusive upper bound for search. Defaults to len(series).

    Returns:
        The iloc of the first window in the stable run, or None if not found.
    """
    arr = np.asarray(series) if not isinstance(series, np.ndarray) else series
    n = len(arr)
    if end_iloc is None:
        end_iloc = n
    end_iloc = min(end_iloc, n)

    run_start = None
    run_len = 0

    for i in range(start_iloc, end_iloc):
        if arr[i] == target_label:
            if run_start is None:
                run_start = i
            run_len += 1
            if run_len >= n_stable:
                return run_start
        else:
            run_start = None
            run_len = 0

    return None


def analyze_transitions(
    df: pd.DataFrame,
    time_col: str = "time",
    true_col: str = "true_label",
    pred_col: str = "pred_label",
    n_stable: int = 3,
    threshold_sec: float = 10.0,
    min_windows_remaining: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Detect all ground-truth transitions in df and measure the detection delay
    for each one.  Operates on a SINGLE run's time-sorted DataFrame.

    For each GT transition at position i:
      1. GT transition time = df[time_col].iloc[i].
      2. Stable detection is searched from position i, bounded by the next GT
         transition index (exclusive). This prevents a correct detection in a
         later epoch from being credited to an earlier transition.
      3. detection_delay_sec = detection_time - gt_time  (always >= 0).
      4. If fewer than n_stable windows remain at the run end  → 'boundary'.
      5. If no stable detection within the search bound      → 'missed'.

    Args:
        df: Single-run DataFrame sorted ascending by time_col.
            Must contain [time_col, true_col, pred_col].
        time_col: Window start time column (float, seconds).
        true_col: Ground-truth label column.
        pred_col: Predicted label column.
        n_stable: Consecutive windows required to declare a stable detection.
        threshold_sec: Delays above this value are flagged with is_late=True.
        min_windows_remaining: Transitions within this many windows of the run
            end are flagged 'boundary'. Defaults to n_stable.

    Returns:
        (transitions_df, summary_stats):

        transitions_df columns:
            run_id, transition_idx, from_label, to_label, gt_time,
            detection_time, detection_delay_sec, outcome, is_late, n_stable_used.

        summary_stats keys:
            n_transitions, n_detected, n_missed, n_boundary, detection_rate,
            mean_delay_sec, median_delay_sec, std_delay_sec, max_delay_sec, n_late.
    """
    if min_windows_remaining is None:
        min_windows_remaining = n_stable

    run_id = df["run_id"].iloc[0] if "run_id" in df.columns else "unknown"
    gt_transitions = detect_gt_transitions(df, time_col=time_col, true_col=true_col)
    pred_arr = df[pred_col].to_numpy()
    times = df[time_col].to_numpy()
    n_windows = len(df)

    rows: list[dict] = []

    for k, trans in enumerate(gt_transitions):
        t_idx = trans["transition_idx"]
        next_t_idx = gt_transitions[k + 1]["transition_idx"] if k + 1 < len(gt_transitions) else n_windows

        # Boundary check: not enough windows left to observe stable detection
        if t_idx + min_windows_remaining > n_windows:
            rows.append({
                "run_id": run_id,
                "transition_idx": t_idx,
                "from_label": trans["from_label"],
                "to_label": trans["to_label"],
                "gt_time": trans["gt_time"],
                "detection_time": float("nan"),
                "detection_delay_sec": float("nan"),
                "outcome": "boundary",
                "is_late": False,
                "n_stable_used": n_stable,
            })
            continue

        det_iloc = find_stable_detection(
            pred_arr, t_idx, trans["to_label"], n_stable, end_iloc=next_t_idx
        )

        if det_iloc is None:
            rows.append({
                "run_id": run_id,
                "transition_idx": t_idx,
                "from_label": trans["from_label"],
                "to_label": trans["to_label"],
                "gt_time": trans["gt_time"],
                "detection_time": float("nan"),
                "detection_delay_sec": float("nan"),
                "outcome": "missed",
                "is_late": False,
                "n_stable_used": n_stable,
            })
        else:
            det_time = float(times[det_iloc])
            delay = det_time - trans["gt_time"]
            rows.append({
                "run_id": run_id,
                "transition_idx": t_idx,
                "from_label": trans["from_label"],
                "to_label": trans["to_label"],
                "gt_time": trans["gt_time"],
                "detection_time": det_time,
                "detection_delay_sec": delay,
                "outcome": "detected",
                "is_late": delay > threshold_sec,
                "n_stable_used": n_stable,
            })

    if rows:
        transitions_df = pd.DataFrame(rows)
    else:
        transitions_df = pd.DataFrame(columns=[
            "run_id", "transition_idx", "from_label", "to_label", "gt_time",
            "detection_time", "detection_delay_sec", "outcome", "is_late", "n_stable_used",
        ])

    # Summary statistics
    n_total = len(rows)
    n_detected = int((transitions_df["outcome"] == "detected").sum()) if n_total else 0
    n_missed = int((transitions_df["outcome"] == "missed").sum()) if n_total else 0
    n_boundary = int((transitions_df["outcome"] == "boundary").sum()) if n_total else 0
    effective = n_total - n_boundary
    det_delays = transitions_df.loc[transitions_df["outcome"] == "detected", "detection_delay_sec"]

    summary: dict[str, Any] = {
        "n_transitions": n_total,
        "n_detected": n_detected,
        "n_missed": n_missed,
        "n_boundary": n_boundary,
        "detection_rate": n_detected / effective if effective > 0 else float("nan"),
        "mean_delay_sec": float(det_delays.mean()) if len(det_delays) else float("nan"),
        "median_delay_sec": float(det_delays.median()) if len(det_delays) else float("nan"),
        "std_delay_sec": float(det_delays.std()) if len(det_delays) else float("nan"),
        "max_delay_sec": float(det_delays.max()) if len(det_delays) else float("nan"),
        "n_late": int(transitions_df.get("is_late", pd.Series(dtype=bool)).sum()),
    }

    return transitions_df, summary


def compare_models(
    df: pd.DataFrame,
    predictions_dict: dict[str, pd.Series],
    time_col: str = "t_start",
    true_col: str = "label",
    run_col: str = "run_id",
    n_stable: int = 3,
    threshold_sec: float = 10.0,
) -> pd.DataFrame:
    """
    Run analyze_transitions for multiple models and aggregate results.

    Args:
        df: Feature DataFrame with metadata (run_col, time_col, true_col).
        predictions_dict: Dict mapping model name → pd.Series of predicted
                          label strings, index-aligned to df.
        time_col: Window start time column.
        true_col: Ground-truth label column.
        run_col: Run identifier column.
        n_stable: Passed to analyze_transitions.
        threshold_sec: Passed to analyze_transitions.

    Returns:
        Combined transitions DataFrame with all analyze_transitions columns
        plus a 'model' column.
    """
    run_ids = sorted(df[run_col].unique().tolist())
    all_rows: list[pd.DataFrame] = []

    for model_name, pred_series in predictions_dict.items():
        for run_id in run_ids:
            run_mask = df[run_col].astype(str) == str(run_id)
            run_df = (
                df.loc[run_mask]
                .sort_values(time_col)
                .assign(
                    time=lambda x: x[time_col],
                    true_label=lambda x: x[true_col],
                    pred_label=pred_series.loc[run_mask].values,
                )
                .reset_index(drop=True)
            )

            t_df, _ = analyze_transitions(
                run_df,
                time_col="time",
                true_col="true_label",
                pred_col="pred_label",
                n_stable=n_stable,
                threshold_sec=threshold_sec,
            )

            if not t_df.empty:
                t_df.insert(0, "model", model_name)
                all_rows.append(t_df)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)

    return pd.DataFrame(columns=[
        "model", "run_id", "transition_idx", "from_label", "to_label", "gt_time",
        "detection_time", "detection_delay_sec", "outcome", "is_late", "n_stable_used",
    ])


# ===========================================================================
# VISUALIZATION
# ===========================================================================

def plot_transition_timeline(
    df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    time_col: str = "time",
    true_col: str = "true_label",
    pred_col: str = "pred_label",
    run_id: str | None = None,
    ax: plt.Axes | None = None,
    label_colors: dict | None = None,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot ground-truth vs predicted terrain label over time for a single run.

    Layout:
      - Top band: ground-truth label (color-coded).
      - Bottom band: predicted label (color-coded).
      - Dashed verticals: GT transition times.
      - Solid verticals: detection times (green=on time, orange=late, red=missed).
      - Delay annotations above each detection line.

    Args:
        df: Single-run, time-sorted DataFrame with [time_col, true_col, pred_col].
        transitions_df: Output of analyze_transitions for this run.
        time_col: Time column name.
        true_col: Ground-truth label column.
        pred_col: Predicted label column.
        run_id: Used in the plot title; inferred from df['run_id'] if None.
        ax: Existing Axes to draw on. If None, a new Figure is created.
        label_colors: Dict mapping label → color. Defaults to TERRAIN_COLORS.
        title: Optional title override.

    Returns:
        matplotlib Figure.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    colors = label_colors or TERRAIN_COLORS
    all_labels = sorted(set(df[true_col].dropna().unique()) | set(df[pred_col].dropna().unique()))

    # Assign color to any label not in TERRAIN_COLORS
    palette = sns.color_palette("husl", len(all_labels))
    for i, lbl in enumerate(all_labels):
        if lbl not in colors:
            colors[lbl] = palette[i]

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3), dpi=150)
    else:
        fig = ax.get_figure()

    times = df[time_col].to_numpy()
    t_min, t_max = times[0], times[-1]

    # Draw color bands for GT (top half) and pred (bottom half)
    band_height = 0.45
    for col, y_center, label_name in [
        (true_col, 0.75, "GT"),
        (pred_col, 0.25, "Pred"),
    ]:
        labels_arr = df[col].to_numpy()
        # Draw a rectangle per window
        for i, lbl in enumerate(labels_arr):
            if pd.isna(lbl):
                continue
            t0 = times[i]
            t1 = times[i + 1] if i + 1 < len(times) else t0 + 1.0
            rect = mpatches.FancyArrow(
                x=t0, y=y_center - band_height / 2,
                dx=t1 - t0, dy=0,
                width=band_height,
                head_width=0, head_length=0,
                color=colors.get(str(lbl), "#aaaaaa"),
                alpha=0.85,
                linewidth=0,
            )
            # Use a simpler fill_between approach instead
            ax.fill_betweenx(
                [y_center - band_height / 2, y_center + band_height / 2],
                t0, t1,
                color=colors.get(str(lbl), "#aaaaaa"),
                alpha=0.85,
                linewidth=0,
            )

    # GT transition lines (dashed)
    for _, row in transitions_df.iterrows():
        ax.axvline(row["gt_time"], color="black", linestyle="--", linewidth=1.5,
                   alpha=0.8, zorder=5)

    # Detection lines (solid)
    for _, row in transitions_df.iterrows():
        if row["outcome"] == "detected":
            c = "orange" if row["is_late"] else "#2ca02c"
            ax.axvline(row["detection_time"], color=c, linestyle="-", linewidth=2.0,
                       alpha=0.9, zorder=6)
            delay_sec = row["detection_delay_sec"]
            ax.text(
                row["detection_time"] + 0.05 * (t_max - t_min) * 0.01,
                0.95,
                f"+{delay_sec:.1f}s",
                transform=ax.get_xaxis_transform(),
                fontsize=7,
                color=c,
                va="top",
                zorder=7,
            )
        elif row["outcome"] == "missed":
            ax.axvline(row["gt_time"], color="red", linestyle=":", linewidth=2.0,
                       alpha=0.7, zorder=6)

    # Y-axis labels
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(["Predicted", "Ground Truth"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlim(t_min, t_max)
    ax.set_xlabel("Time (s)", fontsize=9)

    run_label = run_id or (df["run_id"].iloc[0] if "run_id" in df.columns else "")
    ax.set_title(title or f"Terrain Labels Over Time — {run_label}", fontsize=10)

    # Legend: terrain colors
    legend_patches = [
        mpatches.Patch(color=colors.get(lbl, "#aaaaaa"), label=lbl)
        for lbl in LABEL_ORDER if lbl in all_labels
    ]
    legend_patches += [
        mpatches.Patch(color="black", label="GT transition", linestyle="--", fill=False),
        mpatches.Patch(color="#2ca02c", label="Detection (on time)"),
        mpatches.Patch(color="orange", label="Detection (late)"),
        mpatches.Patch(color="red", label="Missed"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
              ncol=2, framealpha=0.7)

    return fig


def plot_detection_delay_histogram(
    transitions_df: pd.DataFrame,
    model_col: str = "model",
    delay_col: str = "detection_delay_sec",
    outcome_col: str = "outcome",
    bins: int = 15,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Plot detection delay distributions as overlapping histograms, one series per model.

    Shows only 'detected' transitions. Missed and boundary counts are annotated
    as text. Models are colour-coded by the husl palette.

    Args:
        transitions_df: Combined DataFrame from compare_models.
        model_col: Column identifying the model.
        delay_col: Detection delay column (seconds).
        outcome_col: Outcome column.
        bins: Number of histogram bins.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    models = transitions_df[model_col].unique().tolist()
    palette = sns.color_palette("husl", len(models))
    color_map = dict(zip(models, palette))

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150)

    # Left panel: overlapping histograms
    ax_hist = axes[0]
    max_delay = transitions_df.loc[transitions_df[outcome_col] == "detected", delay_col].max()
    bin_edges = np.linspace(0, max_delay * 1.05 if not np.isnan(max_delay) else 30, bins + 1)

    for model_name in models:
        sub = transitions_df[
            (transitions_df[model_col] == model_name) &
            (transitions_df[outcome_col] == "detected")
        ][delay_col].dropna()
        ax_hist.hist(sub, bins=bin_edges, alpha=0.55, label=model_name,
                     color=color_map[model_name], edgecolor="white")

    ax_hist.set_xlabel("Detection Delay (s)", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.set_title("Detection Delay Distribution", fontsize=11)
    ax_hist.legend(fontsize=9)

    # Right panel: mean delay + detection rate bar chart
    ax_bar = axes[1]
    x = np.arange(len(models))
    width = 0.35

    det_rates, mean_delays = [], []
    for model_name in models:
        sub = transitions_df[transitions_df[model_col] == model_name]
        n_det = int((sub[outcome_col] == "detected").sum())
        n_eff = int((sub[outcome_col] != "boundary").sum())
        det_rates.append(n_det / n_eff if n_eff > 0 else 0.0)
        delays = sub.loc[sub[outcome_col] == "detected", delay_col].dropna()
        mean_delays.append(float(delays.mean()) if len(delays) else 0.0)

    bars1 = ax_bar.bar(x - width / 2, det_rates, width, label="Detection Rate",
                       color=[color_map[m] for m in models], alpha=0.8)
    ax2 = ax_bar.twinx()
    bars2 = ax2.bar(x + width / 2, mean_delays, width, label="Mean Delay (s)",
                    color=[color_map[m] for m in models], alpha=0.5, hatch="//")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax_bar.set_ylabel("Detection Rate", fontsize=10)
    ax_bar.set_ylim(0, 1.1)
    ax2.set_ylabel("Mean Detection Delay (s)", fontsize=10)
    ax_bar.set_title("Detection Rate & Mean Delay by Model", fontsize=11)

    # Combined legend
    handles = [bars1[0], bars2[0]]
    labels_leg = ["Detection Rate", "Mean Delay (s)"]
    ax_bar.legend(handles, labels_leg, fontsize=9, loc="upper right")

    # Annotate missed/boundary counts
    y_pos = 1.05
    for i, model_name in enumerate(models):
        sub = transitions_df[transitions_df[model_col] == model_name]
        n_miss = int((sub[outcome_col] == "missed").sum())
        n_bnd = int((sub[outcome_col] == "boundary").sum())
        ax_bar.text(i, y_pos, f"miss={n_miss}\nbnd={n_bnd}",
                    ha="center", va="bottom", fontsize=7, color="gray")

    fig.tight_layout()
    return fig
