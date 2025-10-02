#!/usr/bin/env python3
"""
PART 3: PREDICTION AND VISUALIZATION
- Loads the pre-trained XGBoost model.
- Fetches recent TESS data for 10 random stars.
- Applies the SAME preprocessing and feature engineering as the training script.
- Predicts transit probabilities on the new data.
- Visualizes the light curve and highlights detected transits.
"""

import numpy as np
import lightkurve as lk
import pickle
import os
import gc
from scipy.ndimage import median_filter
from scipy.stats import skew
import matplotlib.pyplot as plt
from astroquery.mast import Observations
from datetime import datetime, timedelta

# ============================================================================
# STEP 1: LOAD THE TRAINED MODEL
# ============================================================================
print("="*70)
print("STEP 1: Loading Pre-trained Exoplanet Detection Model")
print("="*70)

SAVE_PATH = '/content/drive/MyDrive/NASA_SpaceApps_2025/'
model_path = os.path.join(SAVE_PATH, "exoplanet_detector_v4.pkl")

with open(model_path, "rb") as f:
    production_model = pickle.load(f)

model = production_model['xgboost_classifier']
config = production_model['config']
WINDOW = config['window_size'] # Get window size from the model's config

print(f"✅ Model loaded successfully. Using a window size of {WINDOW}.")

# ============================================================================
# STEP 2: FIND 10 RECENT STARS FROM TESS
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Finding 10 Recent Stars from the TESS Mission")
print("="*70)

# Define the time range for "recent" (e.g., last 30 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

print(f"Searching for TESS observations from the last 30 days...")
obs_table = Observations.query_criteria(
    obs_collection="TESS",
    dataproduct_type="timeseries",
    calib_level=3, # SPOC-processed light curves
    t_max=[start_date.timestamp(), end_date.timestamp()] # Observation end time
)

# Get unique target names and select 10 at random
unique_targets = np.unique(obs_table['target_name'])
if len(unique_targets) > 10:
    random_targets = np.random.choice(unique_targets, 10, replace=False)
else:
    random_targets = unique_targets

print(f"✅ Found {len(unique_targets)} unique targets. Selected 10 random ones to analyze:")
for target in random_targets:
    print(f"  - {target}")

# ============================================================================
# STEP 3: PROCESS, PREDICT, AND VISUALIZE EACH STAR
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Processing, Predicting, and Visualizing Light Curves")
print("="*70)

for star_name in random_targets:
    print(f"\n--- Analyzing {star_name} ---")
    try:
        # 1. Download Data
        print("  Downloading light curve data...")
        search = lk.search_lightcurve(star_name, author="SPOC", exptime=120)
        lc = search.download().remove_nans()

        # 2. Apply THE SAME Preprocessing
        print("  Applying preprocessing...")
        normalized = lc.flux.value / np.nanmedian(lc.flux.value)
        detrended = normalized / median_filter(normalized, size=1001)

        # 3. Apply THE SAME Feature Engineering
        print("  Engineering features...")
        new_features = []
        for j in range(WINDOW, len(detrended) - WINDOW):
            window_slice = detrended[j-WINDOW : j+WINDOW]
            features = [
                detrended[j], np.mean(window_slice), detrended[j] - np.mean(window_slice),
                np.min(window_slice), detrended[j] / np.mean(window_slice),
                np.std(window_slice), skew(window_slice)
            ]
            new_features.append(features)
        X_new = np.array(new_features, dtype=np.float32)

        # 4. Make Predictions
        print("  Predicting transits with the trained model...")
        y_proba = model.predict_proba(X_new)[:, 1]

        # 5. Visualize the Results
        print("  Generating plot...")
        # We need to align the predictions with the original time data
        # Predictions start at index WINDOW, so we slice the time and flux arrays
        time_axis = lc.time.value[WINDOW : len(detrended) - WINDOW]
        flux_axis = detrended[WINDOW : len(detrended) - WINDOW]

        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Plot the full light curve
        ax.plot(time_axis, flux_axis, '-', c='gray', label='Detrended Flux', linewidth=0.5)

        # Find points where the model predicted a transit with >50% probability
        transit_indices = np.where(y_proba > 0.5)[0]

        # Plot only the detected transit points on top
        ax.plot(time_axis[transit_indices], flux_axis[transit_indices], 'r.', label='Detected Transit (Prob > 0.5)')

        ax.set_title(f"Transit Detection for {star_name}", fontsize=16)
        ax.set_xlabel("Time (BJD)")
        ax.set_ylabel("Normalized Flux")
        ax.legend()
        plt.show()

        # Clean up memory
        del lc, X_new, y_proba
        gc.collect()

    except Exception as e:
        print(f"  Could not process {star_name}. Error: {e}")
        continue


        
