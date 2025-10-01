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
from scipy.stats import skew # We need this for a powerful feature

# Suppress minor warnings
warnings.filterwarnings("ignore")

def print_header(title, char="="):
    """Prints a formatted header."""
    print("\n" + char * 70)
    print(f"  {title.upper()}")
    print(char * 70)

# --- STEP 1: LOAD DATA AND EXTRACT TO NUMPY ---
print_header("Step 1: Loading Data and Extracting to NumPy")
TARGET_STAR = "pi Mensae"
lc_raw = lk.search_lightcurve(TARGET_STAR, author="SPOC").download_all()[0]

nan_mask = ~np.isnan(lc_raw.flux.value)
time = lc_raw.time.value[nan_mask]
flux = lc_raw.flux.value[nan_mask]
print("Data successfully loaded and extracted into NumPy arrays.")

# --- STEP 2: PREPROCESSING ---
print_header("Step 2: Preprocessing Data")
median_flux = np.nanmedian(flux)
normalized_flux = flux / median_flux

from scipy.signal import medfilt
trend = medfilt(normalized_flux, kernel_size=401)
flattened_flux = normalized_flux - trend
print("Normalization and flattening complete.")

# --- STEP 3: ADVANCED FEATURE ENGINEERING ---
print_header("Step 3: Advanced Statistical Feature Engineering")
WINDOW_SIZE = 128 # A slightly larger window is better for statistical features
known_period = 6.27
phase = (time % known_period) / known_period

X, y = [], []
print(f"Creating statistical features for each window of size {WINDOW_SIZE}...")
for i in range(len(flattened_flux) - WINDOW_SIZE):
    # ================================ THE CRITICAL FIX ================================
    # Instead of raw flux, we calculate powerful statistical features for each window.
    window = flattened_flux[i : i + WINDOW_SIZE]
    
    features = [
        np.mean(window),      # The average flux in the window
        np.std(window),       # The amount of noise/scatter
        np.min(window),       # The lowest point (key for finding dips)
        np.max(window),       # The highest point
        skew(window),         # A measure of asymmetry; dips are negatively skewed
        np.ptp(window)        # The range (max - min)
    ]
    X.append(features)
    # ==================================================================================
    
    center_phase = phase[i + int(WINDOW_SIZE / 2)]
    # We use a very tight phase gate for labeling to ensure high-quality labels
    y.append(1 if 0.99 < center_phase or center_phase < 0.01 else 0)

X = np.array(X)
y = np.array(y)
print(f"Created {X.shape[0]} samples with {X.shape[1]} features each. Found {np.sum(y)} 'Transit' samples.")

# --- STEP 4: TRAIN THE BALANCED CLASSIFIER ---
print_header("Step 4: Training the BALANCED Classifier on Rich Features")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# A more powerful model configuration is needed for these features
model = RandomForestClassifier(
    n_estimators=200,          # More trees for a more complex decision
    max_depth=10,              # Prevents overfitting
    min_samples_leaf=5,        # Smooths the model
    random_state=42,
    class_weight='balanced_subsample', # A more advanced balancing method
    n_jobs=-1
)

print("\nTraining the final Random Forest model...")
model.fit(X_train, y_train)
print("Training complete.")

# --- STEP 5: EVALUATE THE FINAL, WORKING MODEL ---
print_header("Step 5: Evaluating the Final Model")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions, target_names=["No Transit", "Transit"]))

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Predicted No Transit", "Predicted Transit"],
            yticklabels=["Actual No Transit", "Actual Transit"])
plt.title("Confusion Matrix for the Final, Feature-Engineered Model")
plt.savefig("confusion_matrix.png")
print("\nSaved final Confusion Matrix plot to 'confusion_matrix.png'.")
plt.close()

# --- STEP 6: SAVE THE FINAL MODEL ---
print_header("Step 6: Saving the Final Trained Model")
model_filename = "transit_classifier.pkl"
# Note: The app will now need to calculate these features, not use raw flux.
model_data = {"model": model, "window_size": WINDOW_SIZE}
with open(model_filename, "wb") as f:
    pickle.dump(model_data, f)
print(f"The final, working model has been saved to '{model_filename}'.")

print_header("Workflow Complete", char="*")
