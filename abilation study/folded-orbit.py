#!/usr/bin/env python3
"""
PART 1B: PREPROCESSING WITH FOLDING
- Fetches stars with known transiting planets and their precise orbital parameters.
- Uses the known planet period and transit time to create highly accurate transit labels.
- This replaces the anomaly detection method for a more robust "ground truth" dataset.
- Saves the final features and labels to a new HDF5 file.
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import numpy as np
import pandas as pd
import lightkurve as lk
from scipy.ndimage import median_filter
from scipy.stats import skew
import warnings
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import os
import gc
import psutil
import h5py
from datetime import datetime

warnings.filterwarnings("ignore")

def print_header(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

# ============================================================================
# STEP 1: STAR SELECTION (MODIFIED TO GET PLANET PARAMETERS)
# ============================================================================
print_header("STEP 1: Discovering TESS Hosts with Known Planet Parameters")

# We now select pl_orbper (orbital period), pl_tranmid (transit midpoint), and pl_trandur (transit duration)
planets_df = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="hostname, pl_orbper, pl_tranmid, pl_trandur",
    where="tran_flag=1 and sy_tmag is not null and pl_orbper is not null and pl_tranmid is not null and pl_trandur is not null"
).to_pandas().dropna().drop_duplicates(subset=['hostname']) # Removed sort_values('sy_tmag') as it's not selected

available_stars = {}
print("Checking for available 2-minute cadence data...")
# We check the first 100 brightest stars to find at least 35 with good data
for idx, row in planets_df.head(100).iterrows():
    try:
        if row['hostname'] not in available_stars:
            search = lk.search_lightcurve(row['hostname'], author="SPOC", exptime=120)
            if len(search) > 0:
                print(f"  ✓ {row['hostname']:<20} - Data found")
                available_stars[row['hostname']] = {
                    'period': row['pl_orbper'],
                    'transit_time': row['pl_tranmid'],
                    'duration': row['pl_trandur'] / 24.0 # Convert duration from hours to days for lightkurve
                }
    except Exception:
        continue
    # Aim for a solid dataset of ~35 stars
    if len(available_stars) >= 35:
        break

print(f"\nFinal selection: {len(available_stars)} stars with known parameters.")

# ============================================================================
# STEP 2: FEATURE EXTRACTION & LABELING WITH FOLDING
# ============================================================================
print_header("STEP 2: Extracting Features and Labeling via Folding")

SAVE_PATH = '/content/drive/MyDrive/NASA_SpaceApps_2025/'
TEMP_FEATURES_PATH = os.path.join(SAVE_PATH, 'temp_features_folded')
os.makedirs(TEMP_FEATURES_PATH, exist_ok=True)
WINDOW = 32

for star_name, params in available_stars.items():
    star_file_path = os.path.join(TEMP_FEATURES_PATH, f'{star_name.replace(" ", "_")}.npz')
    if os.path.exists(star_file_path):
        print(f"⏭️  Skipping {star_name} (already processed)")
        continue

    print(f"\n🔬 Processing {star_name}...")
    try:
        search = lk.search_lightcurve(star_name, author="SPOC", exptime=120)
        lc = search.download_all().stitch().remove_nans()

        # Preprocessing (Normalization and Detrending)
        normalized = lc.flux.value / np.nanmedian(lc.flux.value)
        detrended = normalized / median_filter(normalized, size=1001)

        # *** NEW LABELING METHOD USING KNOWN PARAMETERS ***
        # Create a boolean mask that is True during a known transit
        transit_mask = lc.create_transit_mask(
            period=params['period'],
            transit_time=params['transit_time'],
            duration=params['duration']
        )
        y_labels = transit_mask.astype('int8')

        # Feature Engineering (sliding window, same as before)
        star_features = []
        for j in range(WINDOW, len(detrended) - WINDOW):
            window = detrended[j-WINDOW:j+WINDOW]
            features = [
                detrended[j], np.mean(window), detrended[j] - np.mean(window),
                np.min(window), detrended[j] / np.mean(window),
                np.std(window), skew(window)
            ]
            star_features.append(features)

        # Convert to numpy arrays
        star_features_np = np.array(star_features, dtype=np.float32)
        # Align labels with the features created by the sliding window
        star_labels_np = y_labels[WINDOW:-WINDOW]

        # Save both features and labels together in a compressed .npz file
        np.savez_compressed(
            star_file_path,
            features=star_features_np,
            labels=star_labels_np
        )
        print(f"  → Extracted and saved {len(star_features_np):,} samples to disk.")
        del lc, normalized, detrended, star_features_np, star_labels_np
        gc.collect()

    except Exception as e:
        print(f"  ✗ Failed to process {star_name}: {e}")
        continue

# ============================================================================
# STEP 3: COMBINE DATA INTO FINAL HDF5 FILE
# ============================================================================
print_header("STEP 3: Combining Data into Final HDF5 Dataset")

feature_files = [os.path.join(TEMP_FEATURES_PATH, f) for f in os.listdir(TEMP_FEATURES_PATH) if f.endswith('.npz')]
total_samples = 0
num_features = 0
star_id_map = []

print("Pass 1: Calculating total dataset size...")
for f_path in feature_files:
    with np.load(f_path) as data:
        n_samples, n_features = data['features'].shape
        total_samples += n_samples
        if num_features == 0: num_features = n_features
        star_name = os.path.basename(f_path).replace('.npz', '').replace('_', ' ')
        star_id_map.extend([star_name] * n_samples)

print(f"Total samples to combine: {total_samples:,}")
print(f"Number of features: {num_features}")

hdf5_path = os.path.join(SAVE_PATH, 'preprocessed_transit_data_FOLDED_v1.h5')
print(f"\nPass 2: Creating final HDF5 dataset at {hdf5_path}")

with h5py.File(hdf5_path, 'w') as hf:
    X_ds = hf.create_dataset('X', shape=(total_samples, num_features), dtype='float32')
    y_ds = hf.create_dataset('y', shape=(total_samples,), dtype='int8')
    star_ids_ds = hf.create_dataset('star_ids', data=[s.encode('utf-8') for s in star_id_map])

    current_row = 0
    for f_path in feature_files:
        with np.load(f_path) as data:
            features_chunk = data['features']
            labels_chunk = data['labels']
            num_rows = features_chunk.shape[0]

            X_ds[current_row : current_row + num_rows, :] = features_chunk
            y_ds[current_row : current_row + num_rows] = labels_chunk

            print(f"  - Combined {os.path.basename(f_path)} ({num_rows:,} samples)")
            current_row += num_rows

    hf.attrs['feature_names'] = ['flux', 'local_mean', 'deviation', 'min', 'ratio', 'std', 'skew']
    hf.attrs['window_size'] = WINDOW
    hf.attrs['creation_date'] = datetime.now().isoformat()

print("\n✅ New accurately-labeled dataset created successfully!")
