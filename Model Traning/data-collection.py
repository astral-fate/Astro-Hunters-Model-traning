#!/usr/bin/env python3
"""
PART 1: OFFLINE PREPROCESSING (v5 - ON-DISK STRATEGY)
- Processes each star and saves its features directly to disk.
- Never accumulates the full dataset in RAM.
- Uses HDF5 for the final dataset, allowing for out-of-core (larger than RAM) processing.
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True) # Added force_remount for convenience

import numpy as np
import pandas as pd
import lightkurve as lk
from sklearn.ensemble import IsolationForest
from scipy.ndimage import median_filter
from scipy.stats import skew
import pickle
import warnings
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import os
import gc
import psutil
import json
from datetime import datetime
import h5py # Import h5py for handling large datasets

warnings.filterwarnings("ignore")

# ============================================================================
# UTILITIES (Memory and Headers - Unchanged)
# ============================================================================

def get_memory_usage():
    """Return current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def print_memory_status(label=""):
    """Print current memory usage"""
    mem_gb = get_memory_usage()
    mem_percent = psutil.virtual_memory().percent
    print(f"  💾 Memory: {mem_gb:.2f} GB ({mem_percent:.1f}%) {label}")

def force_garbage_collection():
    """Aggressively free memory"""
    gc.collect()

def print_header(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

# ============================================================================
# STEP 1: STAR SELECTION (Unchanged)
# ============================================================================
print_header("STEP 1: Discovering a Diverse Set of TESS Exoplanet Hosts")

# This part is efficient and doesn't need changes.
# ... (Your existing star selection code goes here) ...
planets = NasaExoplanetArchive.query_criteria(
    table="pscomppars", select="hostname,pl_orbper,tran_flag,sy_tmag",
    where="tran_flag=1 and sy_tmag is not null"
).to_pandas().drop_duplicates(subset=['hostname']).sort_values('sy_tmag')
manual_add_stars = {'LHS 3844': 0.46, 'TOI-700': 9.98, 'WASP-126': 3.84, 'pi Mensae': 6.27}
available_stars = {}
print("Checking for available 2-minute cadence data...")
for idx, row in planets.head(200).iterrows():
    try:
        if row['hostname'] not in available_stars:
            search = lk.search_lightcurve(row['hostname'], author="SPOC", exptime=120)
            if len(search) > 0:
                print(f"  ✓ {row['hostname']:<20} - Data found")
                available_stars[row['hostname']] = row['pl_orbper']
    except Exception:
        continue
    if len(available_stars) >= 35:
        break
for star_name, period in manual_add_stars.items():
    if star_name not in available_stars:
        try:
            search = lk.search_lightcurve(star_name, author="SPOC", exptime=120)
            if len(search) > 0:
                print(f"  ✓ {star_name:<20} - Data found (Manual)")
                available_stars[star_name] = period
        except:
            print(f"  ✗ {star_name:<20} - Not found")
print(f"\nFinal selection: {len(available_stars)} stars")
print_memory_status("after star selection")


# ============================================================================
# STEP 2: FEATURE EXTRACTION (MODIFIED TO SAVE-TO-DISK)
# ============================================================================
print_header("STEP 2: Extracting Features and Saving to Disk")

SAVE_PATH = '/content/drive/MyDrive/NASA_SpaceApps_2025/'
TEMP_FEATURES_PATH = os.path.join(SAVE_PATH, 'temp_features')
os.makedirs(TEMP_FEATURES_PATH, exist_ok=True)

# State file now just tracks which stars are done.
STATE_FILE = os.path.join(SAVE_PATH, 'processing_state.json') # <-- THIS IS THE LINE TO CHANGE
# CORRECTED LINE
STATE_FILE = os.path.join(SAVE_PATH, 'checkpoints', 'processing_state.json')

# State file now just tracks which stars are done.
# STATE_FILE = os.path.join(SAVE_PATH, 'processing_state.json')

def load_processed_stars():
    if not os.path.exists(STATE_FILE):
        return []
    with open(STATE_FILE, 'r') as f:
        return json.load(f).get('processed_stars', [])

def save_processed_stars(processed_list):
    with open(STATE_FILE, 'w') as f:
        json.dump({'processed_stars': processed_list}, f)

processed_stars = load_processed_stars()
if processed_stars:
    print(f"🔄 Resuming. {len(processed_stars)} stars already processed.")

WINDOW = 32

for star_name, period in available_stars.items():
    star_feature_file = os.path.join(TEMP_FEATURES_PATH, f'{star_name.replace(" ", "_")}.npy')

    # Skip if the feature file for this star already exists
    if star_name in processed_stars and os.path.exists(star_feature_file):
        print(f"⏭️  Skipping {star_name} (already processed)")
        continue

    print(f"\n🔬 Processing {star_name}...")
    print_memory_status("before download")

    try:
        # Download and process data
        search = lk.search_lightcurve(star_name, author="SPOC", exptime=120)
        lc = search.download_all().stitch().remove_nans()
        normalized = lc.flux.value / np.nanmedian(lc.flux.value)
        detrended = normalized / median_filter(normalized, size=1001)

        # Extract features (same logic, but stored locally)
        star_features = []
        for j in range(WINDOW, len(detrended) - WINDOW):
            window = detrended[j-WINDOW:j+WINDOW]
            features = [
                detrended[j], np.mean(window), detrended[j] - np.mean(window),
                np.min(window), detrended[j] / np.mean(window),
                np.std(window), skew(window)
            ]
            star_features.append(features)

        # Convert to numpy array and SAVE TO DISK
        star_features_np = np.array(star_features, dtype=np.float32)
        np.save(star_feature_file, star_features_np)

        print(f"  → Extracted and saved {len(star_features_np):,} samples to disk.")

        # Update and save the list of processed stars
        processed_stars.append(star_name)
        save_processed_stars(processed_stars)

        # Clean up memory aggressively after each star
        del lc, normalized, detrended, star_features, star_features_np
        force_garbage_collection()
        print_memory_status("after processing & cleanup")

    except Exception as e:
        print(f"  ✗ Failed to process {star_name}: {str(e)[:100]}")
        continue

print(f"\n✅ All individual star features saved to disk!")

# ============================================================================
# STEP 3: COMBINE DATA & GENERATE LABELS (NEW ON-DISK METHOD)
# ============================================================================
print_header("STEP 3: Combining Data and Generating Labels from Disk")

feature_files = [os.path.join(TEMP_FEATURES_PATH, f) for f in os.listdir(TEMP_FEATURES_PATH) if f.endswith('.npy')]

# --- Pass 1: Get total size and metadata ---
total_samples = 0
num_features = 0
star_id_map = [] # To store which star each sample belongs to

print("Pass 1: Calculating total dataset size...")
for f_path in feature_files:
    star_name = os.path.basename(f_path).replace('.npy', '').replace('_', ' ')
    data = np.load(f_path, mmap_mode='r') # Read metadata without loading data
    n_samples, n_features = data.shape
    total_samples += n_samples
    if num_features == 0: num_features = n_features
    star_id_map.extend([star_name] * n_samples)

print(f"Total samples to combine: {total_samples:,}")
print(f"Number of features: {num_features}")
print_memory_status("after pass 1")


# --- Pass 2: Create final HDF5 file and populate it ---
hdf5_path = os.path.join(SAVE_PATH, 'preprocessed_transit_data_v5.h5')
print(f"\nPass 2: Creating final HDF5 dataset at {hdf5_path}")

with h5py.File(hdf5_path, 'w') as hf:
    # Create datasets
    X_ds = hf.create_dataset('X', shape=(total_samples, num_features), dtype='float32')
    y_ds = hf.create_dataset('y', shape=(total_samples,), dtype='int8')
    star_ids_ds = hf.create_dataset('star_ids', data=[s.encode('utf-8') for s in star_id_map])

    # --- Fit IsolationForest on a sample of the data ---
    # Loading everything to fit is impossible. So we fit on a random 1% sample.
    print("Fitting Anomaly Detector on a 1% data sample...")
    sample_size = total_samples // 100
    random_indices = np.random.choice(total_samples, size=sample_size, replace=False)
    random_indices.sort() # Sorting helps with read efficiency

    # Load only the data for the random indices
    sample_X = np.empty((sample_size, num_features), dtype=np.float32)
    
    # This is a bit complex, but it efficiently pulls random rows from all files
    current_pos = 0
    sample_idx_pos = 0
    for f_path in feature_files:
        data = np.load(f_path)
        file_len = len(data)
        
        # Find which random indices fall within this file
        indices_in_file = []
        while sample_idx_pos < len(random_indices) and random_indices[sample_idx_pos] < current_pos + file_len:
            indices_in_file.append(random_indices[sample_idx_pos] - current_pos)
            sample_idx_pos += 1
            
        if indices_in_file:
            # How many rows we've filled in sample_X so far
            fill_start = len(indices_in_file) - len(indices_in_file) 
            fill_end = fill_start + len(indices_in_file)
            sample_X[fill_start:fill_end] = data[indices_in_file]
        
        current_pos += file_len
        del data
        gc.collect()

    anomaly_detector = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
    anomaly_detector.fit(sample_X)
    del sample_X # free sample memory
    gc.collect()
    print("  ✓ Detector fitted.")

    # --- Populate datasets by reading one file at a time ---
    print("Populating final dataset and predicting labels in chunks...")
    current_row = 0
    for f_path in feature_files:
        data = np.load(f_path)
        num_rows = data.shape[0]

        # Write features to HDF5
        X_ds[current_row : current_row + num_rows, :] = data

        # Predict labels for this chunk and write to HDF5
        labels = (anomaly_detector.predict(data) == -1).astype(np.int8)
        y_ds[current_row : current_row + num_rows] = labels
        
        print(f"  - Processed {os.path.basename(f_path)} ({num_rows:,} samples)")

        current_row += num_rows
        del data, labels # Clean up
        force_garbage_collection()

    # Add metadata to the HDF5 file
    hf.attrs['feature_names'] = ['flux', 'local_mean', 'deviation', 'min', 'ratio', 'std', 'skew']
    hf.attrs['window_size'] = WINDOW
    hf.attrs['creation_date'] = datetime.now().isoformat()

print("✅ HDF5 dataset created successfully!")
print_memory_status("final")

# Optional: Clean up temporary files
# import shutil
# shutil.rmtree(TEMP_FEATURES_PATH)
# os.remove(STATE_FILE)
# print("✓ Temporary files cleaned up.")


