#!/usr/bin/env python3
"""
PART 2B: MODEL TRAINING (v3 - ADVANCED METRICS)
- Loads the accurately-labeled HDF5 file created by the "folding" script.
- Trains the XGBoost model.
- Evaluates with advanced metrics: Confusion Matrix and Precision-Recall Curve.
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                             precision_recall_curve, average_precision_score)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def print_header(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

# ============================================================================
# STEP 1: LOAD PREPROCESSED DATASET (POINTING TO NEW FOLDED DATA)
# ============================================================================
print_header("STEP 1: Loading Preprocessed Dataset (from Folded Labels)")
SAVE_PATH = '/content/drive/MyDrive/NASA_SpaceApps_2025/'
# ⚠️ Point to the NEW HDF5 file created by the folding script
dataset_path = f"{SAVE_PATH}preprocessed_transit_data_FOLDED_v1.h5" 

hf = h5py.File(dataset_path, "r")
X_h5 = hf['X']
y_h5 = hf['y']
star_ids_h5 = hf['star_ids']

print(f"Successfully opened HDF5 file with {len(X_h5):,} samples.")
print(f"Data shape: {X_h5.shape}")

# ============================================================================
# STEP 2: CREATE LEAK-PROOF SPLIT (Unchanged)
# ============================================================================
print_header("STEP 2: Creating Leak-Proof Train & Hold-out Test Sets")

star_ids = np.array([s.decode('utf-8') for s in star_ids_h5[:]])
unique_stars = np.unique(star_ids)
train_val_stars, test_stars = train_test_split(unique_stars, test_size=0.20, random_state=42)

print(f"Training/Validation stars ({len(train_val_stars)}): {', '.join(train_val_stars)}")
print(f"Hold-out Test stars ({len(test_stars)}): {', '.join(test_stars)}")

train_indices = np.where(np.isin(star_ids, train_val_stars))[0]
test_indices = np.where(np.isin(star_ids, test_stars))[0]

print("\nLoading training data into RAM...")
X_train = X_h5[train_indices]
y_train = y_h5[train_indices]

print("Loading hold-out test data into RAM...")
X_test = X_h5[test_indices]
y_test = y_h5[test_indices]

print(f"\nTrain set size: {len(X_train):,}")
print(f"Hold-out Test set size: {len(X_test):,}")

hf.close()

# ============================================================================
# STEP 3: TRAIN AND EVALUATE XGBOOST MODEL (WITH ADVANCED METRICS)
# ============================================================================
print_header("STEP 3: Training and Evaluating XGBoost Classifier")
xgb_model = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    scale_pos_weight=sum(y_train == 0) / sum(y_train == 1), # Dynamically calculate for class balance
    random_state=42, eval_metric='logloss',
    use_label_encoder=False, n_jobs=-1
)

print("Training model...")
xgb_model.fit(X_train, y_train)
print("→ Model training complete.")

print("\nEvaluating on the UNSEEN Hold-out Test Set...")
y_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.5).astype(int)

# --- Standard Performance Report ---
print("\nHold-out Test Set Performance:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Transit']))
print(f"AUC-ROC on Hold-out Test Set: {roc_auc_score(y_test, y_proba):.4f}")

# --- NEW: Confusion Matrix ---
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Transit'],
            yticklabels=['Actual Normal', 'Actual Transit'])
plt.title('Confusion Matrix', fontsize=16)
plt.show()

# --- NEW: Precision-Recall Curve ---
print("\nPrecision-Recall Curve:")
precision, recall, _ = precision_recall_curve(y_test, y_proba)
auprc = average_precision_score(y_test, y_proba)
print(f"Area Under PR Curve (AUPRC): {auprc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f'AUPRC = {auprc:.4f}')
plt.title('Precision-Recall Curve', fontsize=16)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.show()

# ============================================================================
# STEP 4: SAVE PRODUCTION MODEL (with new name)
# ============================================================================
print_header("STEP 4: Saving Production Model")
with h5py.File(dataset_path, 'r') as temp_hf:
    config = {'window_size': temp_hf.attrs['window_size']}

production_model = {
    'xgboost_classifier': xgb_model,
    'config': config,
    'performance': {
        'holdout_test_auc': float(roc_auc_score(y_test, y_proba)),
        'holdout_test_auprc': float(auprc)
    }
}
model_path = f"{SAVE_PATH}exoplanet_detector_FOLDED_v1.pkl" # New model name
with open(model_path, "wb") as f:
    pickle.dump(production_model, f)

print(f"\n→ Production model saved: {model_path}")
print("\nTRAINING PIPELINE COMPLETE.")
