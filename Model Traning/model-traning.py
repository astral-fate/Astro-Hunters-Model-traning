#!/usr/bin/env python3
"""
PART 2: MODEL TRAINING (v2 - HDF5 COMPATIBLE)
- Loads the pre-processed data from a large HDF5 file.
- Performs a memory-efficient, leak-proof train/test split based on star names.
- Trains the XGBoost model and saves the final production-ready file.
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import numpy as np
import h5py  # Use h5py instead of pickle for loading data
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle

def print_header(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

# ============================================================================
# STEP 1: LOAD PREPROCESSED DATASET (MODIFIED FOR HDF5)
# ============================================================================
print_header("STEP 1: Loading Preprocessed Dataset from HDF5 File")
SAVE_PATH = '/content/drive/MyDrive/NASA_SpaceApps_2025/'
# ⚠️ Point to your new .h5 file
dataset_path = f"{SAVE_PATH}preprocessed_transit_data_v5.h5" 

# Open the HDF5 file in read mode
hf = h5py.File(dataset_path, "r")

# Get handles to the datasets. This does NOT load them into memory yet!
X_h5 = hf['X']
y_h5 = hf['y']
star_ids_h5 = hf['star_ids']

print(f"Successfully opened HDF5 file with {len(X_h5):,} samples.")
print(f"Data shape: {X_h5.shape}")

# ============================================================================
# STEP 2: CREATE LEAK-PROOF SPLIT (MEMORY-EFFICIENT METHOD)
# ============================================================================
print_header("STEP 2: Creating Leak-Proof Train & Hold-out Test Sets")

# Load only the star_ids into memory for the split logic
# Decode from bytes (UTF-8) to strings
star_ids = np.array([s.decode('utf-8') for s in star_ids_h5[:]])

unique_stars = np.unique(star_ids)
train_val_stars, test_stars = train_test_split(unique_stars, test_size=0.20, random_state=42)

print(f"Training/Validation stars ({len(train_val_stars)}): {', '.join(train_val_stars)}")
print(f"Hold-out Test stars ({len(test_stars)}): {', '.join(test_stars)}")

# Instead of creating a giant boolean mask, get the integer indices
print("\nCalculating indices for train and test sets...")
train_indices = np.where(np.isin(star_ids, train_val_stars))[0]
test_indices = np.where(np.isin(star_ids, test_stars))[0]

# NOW, use the indices to load only the necessary data into memory
print("Loading training data into RAM...")
X_train = X_h5[train_indices]
y_train = y_h5[train_indices]

print("Loading hold-out test data into RAM...")
X_test = X_h5[test_indices]
y_test = y_h5[test_indices]

print(f"\nTrain set size: {len(X_train):,}")
print(f"Hold-out Test set size: {len(X_test):,}")

# Close the HDF5 file now that we have what we need in memory
hf.close()

# ============================================================================
# STEP 3: TRAIN AND EVALUATE XGBOOST MODEL (Unchanged)
# ============================================================================
print_header("STEP 3: Training and Evaluating XGBoost Classifier")
# This section remains identical because X_train, y_train etc. are now standard NumPy arrays
xgb_model = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    scale_pos_weight=25, random_state=42, eval_metric='logloss',
    use_label_encoder=False, n_jobs=-1
)

print("Training model...")
xgb_model.fit(X_train, y_train)
print("→ Model training complete.")

print("\nEvaluating on the UNSEEN Hold-out Test Set...")
y_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.5).astype(int)

print("\nHold-out Test Set Performance:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Transit']))
print(f"AUC-ROC on Hold-out Test Set: {roc_auc_score(y_test, y_proba):.4f}")

# ============================================================================
# STEP 4: SAVE PRODUCTION MODEL
# ============================================================================
print_header("STEP 4: Saving Production Model")
# We re-open the file just to get the config metadata from its attributes
with h5py.File(dataset_path, 'r') as temp_hf:
    config = {'window_size': temp_hf.attrs['window_size']}

production_model = {
    'xgboost_classifier': xgb_model,
    'config': config,
    'performance': {'holdout_test_auc': float(roc_auc_score(y_test, y_proba))}
}
model_path = f"{SAVE_PATH}exoplanet_detector_v4.pkl" # Version up the model
with open(model_path, "wb") as f:
    pickle.dump(production_model, f)

print(f"\n→ Production model saved: {model_path}")
print("\nTRAINING PIPELINE COMPLETE.")
