"""
HCP REST1_LR → Schaefer 200 → Covariance → Logistic Regression
=============================================================

This script:
1. Iterates over HCP subject directories (subject_XXXXX)
2. Loads rfMRI_REST1_LR.nii.gz for each subject
3. Applies the Schaefer 200 atlas using nilearn
4. Extracts regional time series (200 x T)
5. Computes covariance matrices
6. Vectorizes the upper triangular part
7. Trains and evaluates a logistic regression model to predict Sex_Binary

This is a clean, interpretable, and standard neuroimaging ML baseline.
"""

# =========================
# Imports
# =========================

from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm

from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


# =========================
# Configuration
# =========================

# Root directory of your HCP data
HCP_ROOT = Path("/sci/labs/arieljaffe/dan.abergel1/HCP_data")

# Labels file
LABELS_JSON = HCP_ROOT / "model_input" / "imageID_to_labels.json"

# Atlas configuration
N_REGIONS = 200
ATLAS_RESOLUTION_MM = 2

# ML configuration
N_SPLITS = 5
RANDOM_STATE = 42

# Cache configuration
USE_CACHED_SCHAEFER = False  # If True, load precomputed Schaefer time series (.npy) instead of recomputing


# =========================
# Load labels
# =========================

with open(LABELS_JSON, "r") as f:
    labels_dict = json.load(f)


# =========================
# Utility functions
# =========================

def get_rest1_lr_nifti(subject_dir: Path) -> Path | None:
    """
    Return the path to rfMRI_REST1_LR.nii.gz for a given subject directory.

    Expected structure:
    subject_xxxxx/
        MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz
    """
    nifti_path = (
        subject_dir
        / "MNINonLinear"
        / "Results"
        / "rfMRI_REST1_LR"
        / "rfMRI_REST1_LR.nii.gz"
    )

    return nifti_path if nifti_path.exists() else None


def save_schaefer_timeseries(timeseries: np.ndarray, nifti_path: Path) -> Path:
    """
    Save Schaefer time series next to the original NIfTI file.

    The file is saved as:
        rfMRI_REST1_LR_schaefer200.npy

    Args:
        timeseries: array of shape (T, 200)
        nifti_path: path to rfMRI_REST1_LR.nii.gz

    Returns:
        Path to the saved .npy file
    """
    out_path = nifti_path.parent / "rfMRI_REST1_LR_schaefer200.npy"
    np.save(out_path, timeseries.astype(np.float32))
    return out_path


def load_schaefer_timeseries(nifti_path: Path) -> np.ndarray | None:
    """
    Load cached Schaefer time series if it exists.

    Expected file:
        rfMRI_REST1_LR_schaefer200.npy
    """
    schaefer_path = nifti_path.parent / "rfMRI_REST1_LR_schaefer200.npy"
    if schaefer_path.exists():
        return np.load(schaefer_path)
    return None


def compute_covariance(timeseries: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix from regional time series.

    Args:
        timeseries: array of shape (T, N_REGIONS)

    Returns:
        cov: array of shape (N_REGIONS, N_REGIONS)
    """
    # Transpose to (regions, time)
    X = timeseries.T

    # Remove temporal mean
    X = X - X.mean(axis=1, keepdims=True)

    # Covariance
    cov = (X @ X.T) / (X.shape[1] - 1)

    return cov


def vectorize_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """
    Vectorize the upper triangular part of a symmetric matrix (excluding diagonal).
    """
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


# =========================
# Load Schaefer atlas
# =========================

print("Loading Schaefer atlas...")
atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=N_REGIONS,
    resolution_mm=ATLAS_RESOLUTION_MM
)

masker = NiftiLabelsMasker(
    labels_img=atlas.maps,
    standardize=False,       # we standardize later
    detrend=False,
    verbose=0
)


# =========================
# Dataset construction
# =========================

X_list = []
y_list = []

subject_dirs = sorted(HCP_ROOT.glob("subject_*"))

print(f"Found {len(subject_dirs)} subject directories")

for subject_dir in tqdm(subject_dirs):

    subject_id = subject_dir.name.replace("subject_", "")
    scan_key = f"{subject_id}_REST1_LR"

    # Check label availability
    if scan_key not in labels_dict:
        continue

    nifti_path = get_rest1_lr_nifti(subject_dir)
    if nifti_path is None:
        print(f"[MISSING NIFTI] {subject_dir.name}")
        continue

    try:
        # Load cached Schaefer time series if requested and available
        timeseries = None

        if USE_CACHED_SCHAEFER:
            timeseries = load_schaefer_timeseries(nifti_path)

        if timeseries is None:
            # Compute Schaefer time series from NIfTI
            timeseries = masker.fit_transform(nifti_path)

            # Save for future runs
            save_schaefer_timeseries(timeseries, nifti_path)

        # Compute covariance
        cov = compute_covariance(timeseries)

        # Vectorize
        features = vectorize_upper_triangle(cov)

        # Label
        sex = labels_dict[scan_key]["Sex_Binary"]

        X_list.append(features)
        y_list.append(sex)

    except Exception as e:
        print(f"[ERROR] {subject_dir.name}: {e}")
        continue


X = np.stack(X_list)
y = np.array(y_list)

print("===================================")
print(f"Final dataset shape: X = {X.shape}")
print(f"Number of subjects: {len(y)}")
print(f"Number of features: {X.shape[1]}")
print("Class distribution:", np.bincount(y))
print("===================================")


# =========================
# Machine learning
# =========================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("ROC-AUC per fold:", scores)
print(f"Mean ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")