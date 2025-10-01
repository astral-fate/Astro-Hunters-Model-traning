#!/usr/bin/env python3
"""
NASA Space Apps 2025 - Exoplanet Transit Detection
Part 1: Generate Training Labels via Anomaly Detection
"""

# !pip install lightkurve scikit-learn xgboost -q

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import lightkurve as lk
from sklearn.ensemble import IsolationForest
from scipy.ndimage import median_filter
from scipy.stats import skew
import pickle
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAINING_STARS = {
    'HD 219134': 3.09,
    'HD 39091': 6.27,
    '55 Cnc': 0.74,
    'HD 158259': 2.18,
    'HR 858': 3.59,
    'HD 63433': 7.11,
    'WASP-189': 2.72,
    'HD 25463': 7.05,
    'HIP 56998': 6.20,
    'AU Mic': 8.46,
    'TOI-480': 6.87,
    'HD 189733': 2.22,
}

WINDOW = 32
SAVE_PATH = '/content/drive/MyDrive/NASA_SpaceApps_2025/'

import os
os.makedirs(SAVE_PATH, exist_ok=True)

print("="*70)
print("NASA SPACE APPS 2025 - EXOPLANET DETECTION PIPELINE")
print("="*70)

# ============================================================================
# STEP 1: EXTRACT FEATURES FROM ALL STARS
# ============================================================================
print("\n[STEP 1/4] Extracting features from TESS data...")

all_features = []
all_star_ids = []
all_times = []
all_flux = []

for idx, (star_name, period) in enumerate(TRAINING_STARS.items(), 1):
    print(f"\n[{idx}/{len(TRAINING_STARS)}] {star_name}")
    
    try:
        search = lk.search_lightcurve(star_name, author="SPOC", exptime=120)
        lc = search.download_all()[0]
        
        nan_mask = ~np.isnan(lc.flux.value)
        time = lc.time.value[nan_mask]
        flux = lc.flux.value[nan_mask]
        
        # Preprocess
        normalized = flux / np.nanmedian(flux)
        long_trend = median_filter(normalized, size=1001)
        detrended = normalized / long_trend
        
        # Extract features
        for i in range(WINDOW, len(detrended) - WINDOW):
            window = detrended[i-WINDOW:i+WINDOW]
            
            features = [
                detrended[i],                        # Current flux
                np.mean(window),                      # Local mean
                detrended[i] - np.mean(window),      # Deviation
                np.min(window),                       # Min in window
                detrended[i] / np.mean(window),      # Ratio
                np.std(window),                       # Variability
                skew(window),                         # Asymmetry
            ]
            
            all_features.append(features)
            all_star_ids.append(star_name)
            all_times.append(time[i])
            all_flux.append(detrended[i])
        
        print(f"  → {len(detrended) - 2*WINDOW:,} samples extracted")
        
    except Exception as e:
        print(f"  → Error: {str(e)[:50]}")
        continue

X = np.array(all_features)
star_ids = np.array(all_star_ids)
times = np.array(all_times)
flux_values = np.array(all_flux)

print(f"\nTotal samples collected: {len(X):,}")

# ============================================================================
# STEP 2: TRAIN ANOMALY DETECTOR TO GENERATE LABELS
# ============================================================================
print("\n[STEP 2/4] Training Isolation Forest for label generation...")

anomaly_detector = IsolationForest(
    contamination=0.02,  # Expect 2% transits
    random_state=42,
    n_estimators=200,
    max_samples=10000,
    n_jobs=-1,
    verbose=0
)

anomaly_detector.fit(X)
print("  → Anomaly detector trained")

# Generate labels
anomaly_predictions = anomaly_detector.predict(X)
y_labels = (anomaly_predictions == -1).astype(int)  # 1 = transit, 0 = normal

print(f"\nGenerated labels:")
print(f"  Transit points: {np.sum(y_labels):,} ({100*np.sum(y_labels)/len(y_labels):.2f}%)")
print(f"  Normal points: {np.sum(y_labels==0):,} ({100*np.sum(y_labels==0)/len(y_labels):.2f}%)")

# Save labeled dataset
labeled_data = {
    'X': X,
    'y': y_labels,
    'star_ids': star_ids,
    'times': times,
    'flux': flux_values,
    'feature_names': ['flux', 'local_mean', 'deviation', 'min', 'ratio', 'std', 'skew'],
    'config': {
        'window_size': WINDOW,
        'training_stars': TRAINING_STARS
    }
}

with open(f"{SAVE_PATH}labeled_dataset.pkl", "wb") as f:
    pickle.dump(labeled_data, f)

print(f"\n  → Labeled dataset saved to {SAVE_PATH}labeled_dataset.pkl")

# ============================================================================
# STEP 3: TRAIN SUPERVISED MODEL (XGBoost)
# ============================================================================
print("\n[STEP 3/4] Training XGBoost classifier on labeled data...")

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Star-wise split (no data leakage)
unique_stars = np.unique(star_ids)
np.random.seed(42)
shuffled = np.random.permutation(unique_stars)

train_stars = shuffled[:int(0.7*len(unique_stars))]
test_stars = shuffled[int(0.7*len(unique_stars)):]

train_mask = np.isin(star_ids, train_stars)
test_mask = np.isin(star_ids, test_stars)

X_train, y_train = X[train_mask], y_labels[train_mask]
X_test, y_test = X[test_mask], y_labels[test_mask]

print(f"  Train: {len(X_train):,} samples from {len(train_stars)} stars")
print(f"  Test: {len(X_test):,} samples from {len(test_stars)} stars")

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=50,  # Handle imbalance
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_model.fit(X_train, y_train)
print("  → XGBoost training complete")

# Evaluate
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

print("\nTest Set Performance:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Transit']))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")

# ============================================================================
# STEP 4: SAVE PRODUCTION MODEL
# ============================================================================
print("\n[STEP 4/4] Saving production model...")

production_model = {
    'xgboost_classifier': xgb_model,
    'anomaly_detector': anomaly_detector,
    'config': {
        'window_size': WINDOW,
        'feature_names': ['flux', 'local_mean', 'deviation', 'min', 'ratio', 'std', 'skew'],
        'training_stars': list(TRAINING_STARS.keys())
    },
    'performance': {
        'test_auc': float(roc_auc_score(y_test, y_proba)),
        'n_training_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
}

model_path = f"{SAVE_PATH}exoplanet_detector_v1.pkl"
with open(model_path, "wb") as f:
    pickle.dump(production_model, f)

print(f"\n  → Production model saved: {model_path}")
print("\n" + "="*70)
print("PIPELINE COMPLETE - Ready for web deployment")
print("="*70)
