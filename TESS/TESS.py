#!/usr/bin/env python3
import warnings
import pickle
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Suppress minor warnings
warnings.filterwarnings("ignore")

def print_header(title, char="="):
    """Prints a formatted header."""
    print("\n" + char * 70)
    print(f"  {title.upper()}")
    print(char * 70)

# --- STEP 1: LOAD DATA AND CONVERT TO SIMPLE NUMPY ARRAYS ---
print_header("Step 1: Loading Data and Extracting to NumPy")
TARGET_STAR = "pi Mensae"
print(f"Searching for TESS data for '{TARGET_STAR}'...")
search_result = lk.search_lightcurve(TARGET_STAR, author="SPOC")
lc_raw = search_result[search_result.exptime.value == 120].download()

# ================================ THE DEFINITIVE FIX ================================
# Immediately extract the data we need into simple, robust NumPy arrays.
# This avoids all future object-oriented errors with lightkurve.
raw_time = lc_raw.time.value
raw_flux = lc_raw.flux.value

# Remove NaN values which can interfere with processing
nan_mask = ~np.isnan(raw_flux)
time = raw_time[nan_mask]
flux = raw_flux[nan_mask]

print("Data successfully loaded and extracted into NumPy arrays.")
# =====================================================================================

# --- STEP 2: PERFORM ALL PREPROCESSING WITH NUMPY ---
print_header("Step 2: Preprocessing Data using NumPy")

# 1. Normalize the flux
median_flux = np.nanmedian(flux)
normalized_flux = flux / median_flux

# 2. Flatten the light curve (removes stellar variability)
# We will use a simple median filter, which is robust and effective.
from scipy.signal import medfilt
trend = medfilt(normalized_flux, kernel_size=401) # 401 is a common window size
flattened_flux = normalized_flux - trend + 1 # Add 1 to keep it centered at 1.0

# 3. Bin the data to improve signal-to-noise
# We'll reshape the array and take the median of each block of 10.
bin_size = 10
# Trim the array so its length is a multiple of bin_size
trim_size = len(flattened_flux) // bin_size * bin_size
binned_flux = np.nanmedian(flattened_flux[:trim_size].reshape(-1, bin_size), axis=1)
binned_time = np.nanmedian(time[:trim_size].reshape(-1, bin_size), axis=1)
print("Normalization, flattening, and binning complete using robust NumPy methods.")

# --- STEP 3: FEATURE ENGINEERING ON THE CLEAN DATA ---
print_header("Step 3: Feature Engineering")
WINDOW_SIZE = 16
known_period = 6.27
# Calculate phase for the binned time data
phase = (binned_time % known_period) / known_period

X, y = [], []
for i in range(len(binned_flux) - WINDOW_SIZE):
    window_flux = binned_flux[i : i + WINDOW_SIZE]
    X.append(window_flux)
    # Label based on the phase at the center of the window
    center_phase = phase[i + int(WINDOW_SIZE / 2)]
    # A wider phase gate for binned data can be more robust
    y.append(1 if 0.95 < center_phase or center_phase < 0.05 else 0)

X = np.array(X)
y = np.array(y)
print(f"Created {X.shape[0]} samples. Found {np.sum(y)} 'Transit' samples.")

# --- STEP 4: TRAIN THE BALANCED CLASSIFIER ---
print_header("Step 4: Training the BALANCED Classifier")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
model.fit(X_train, y_train)
print("Training complete.")

# --- STEP 5: EVALUATE THE FINAL MODEL ---
print_header("Step 5: Evaluating the Final Model")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions, target_names=["No Transit", "Transit"]))

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Predicted No Transit", "Predicted Transit"],
            yticklabels=["Actual No Transit", "Actual Transit"])
plt.title("Confusion Matrix for the Final, Correct Model")
plt.savefig("confusion_matrix.png")
print("\nSaved final Confusion Matrix plot to 'confusion_matrix.png'.")
plt.close()

# --- STEP 6: SAVE THE FINAL MODEL ---
print_header("Step 6: Saving the Final Trained Model")
model_filename = "transit_classifier.pkl"
model_data = {"model": model, "window_size": 64} # App will still use the original window size
with open(model_filename, "wb") as f:
    pickle.dump(model_data, f)
print(f"The final, working model has been saved to '{model_filename}'.")

print_header("Workflow Complete", char="*")
