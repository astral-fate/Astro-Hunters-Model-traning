

# STEP 1:  Preprocessed Dataset

Successfully opened HDF5 file with 3,501,577 samples.
Data shape: (3501577, 7)


# summary 
<details>

Checking for available 2-minute cadence data...
  ✓ Kepler-334           - Data found
  ✓ Kepler-251           - Data found
  ✓ Kepler-729           - Data found
  ✓ Kepler-687           - Data found
  ✓ Kepler-119           - Data found
  ✓ Kepler-1115          - Data found
  ✓ Kepler-1073          - Data found
  ✓ Kepler-405           - Data found
No data found for target "Kepler-899".
ERROR:lightkurve.search:No data found for target "Kepler-899".
  ✓ Kepler-1255          - Data found
  ✓ KELT-23 A            - Data found
  ✓ Kepler-1447          - Data found
  ✓ TOI-220              - Data found
  ✓ HD 23472             - Data found
  ✓ TOI-5726             - Data found
  ✓ TOI-2285             - Data found
  ✓ Kepler-659           - Data found
  ✓ HD 22946             - Data found
  ✓ Kepler-1118          - Data found
  ✓ Kepler-133           - Data found
  ✓ Kepler-620           - Data found
  ✓ Kepler-92            - Data found
  ✓ HAT-P-51             - Data found
  ✓ Kepler-179           - Data found
  ✓ Kepler-166           - Data found
  ✓ Kepler-595           - Data found
  ✓ Kepler-28            - Data found
  ✓ Kepler-210           - Data found
  ✓ Kepler-782           - Data found
  ✓ Kepler-414           - Data found
  ✓ Kepler-238           - Data found
  ✓ Kepler-664           - Data found
  ✓ Kepler-1123          - Data found
  ✓ Kepler-1098          - Data found
  ✓ HAT-P-49             - Data found
  ✓ XO-4                 - Data found

Final selection: 35 stars with known parameters.


STEP 2: Extracting Features and Labeling via Folding


🔬 Processing Kepler-334...
  → Extracted and saved 64,008 samples to disk.

🔬 Processing Kepler-251...
  → Extracted and saved 36,479 samples to disk.

🔬 Processing Kepler-729...
  → Extracted and saved 46,097 samples to disk.

🔬 Processing Kepler-687...
  → Extracted and saved 34,487 samples to disk.

🔬 Processing Kepler-119...
  → Extracted and saved 74,561 samples to disk.

🔬 Processing Kepler-1115...
  → Extracted and saved 129,801 samples to disk.

🔬 Processing Kepler-1073...
  → Extracted and saved 73,265 samples to disk.

🔬 Processing Kepler-405...
  → Extracted and saved 16,748 samples to disk.

🔬 Processing Kepler-1255...
  → Extracted and saved 36,408 samples to disk.

🔬 Processing KELT-23 A...
  → Extracted and saved 273,305 samples to disk.

🔬 Processing Kepler-1447...
  → Extracted and saved 130,086 samples to disk.

🔬 Processing TOI-220...
  → Extracted and saved 549,879 samples to disk.

🔬 Processing HD 23472...
Warning: 30% (5872/19412) of the cadences will be ignored due to the quality mask (quality_bitmask=17087).
WARNING:lightkurve.utils:Warning: 30% (5872/19412) of the cadences will be ignored due to the quality mask (quality_bitmask=17087).
  → Extracted and saved 207,312 samples to disk.

🔬 Processing TOI-5726...
  → Extracted and saved 77,657 samples to disk.

🔬 Processing TOI-2285...
  → Extracted and saved 151,946 samples to disk.

🔬 Processing Kepler-659...
  → Extracted and saved 45,177 samples to disk.

🔬 Processing HD 22946...
Warning: 30% (5871/19412) of the cadences will be ignored due to the quality mask (quality_bitmask=17087).
WARNING:lightkurve.utils:Warning: 30% (5871/19412) of the cadences will be ignored due to the quality mask (quality_bitmask=17087).
  → Extracted and saved 61,999 samples to disk.

🔬 Processing Kepler-1118...
  → Extracted and saved 34,542 samples to disk.

🔬 Processing Kepler-133...
  → Extracted and saved 125,156 samples to disk.

🔬 Processing Kepler-620...
  → Extracted and saved 81,594 samples to disk.

🔬 Processing Kepler-92...
  → Extracted and saved 163,600 samples to disk.

🔬 Processing HAT-P-51...
  → Extracted and saved 32,719 samples to disk.

🔬 Processing Kepler-179...
  → Extracted and saved 103,581 samples to disk.

🔬 Processing Kepler-166...
  → Extracted and saved 67,300 samples to disk.

🔬 Processing Kepler-595...
  → Extracted and saved 79,579 samples to disk.

🔬 Processing Kepler-28...
  → Extracted and saved 72,630 samples to disk.

🔬 Processing Kepler-210...
  → Extracted and saved 147,859 samples to disk.

🔬 Processing Kepler-782...
  → Extracted and saved 79,002 samples to disk.

🔬 Processing Kepler-414...
  → Extracted and saved 143,642 samples to disk.

🔬 Processing Kepler-238...
  → Extracted and saved 69,506 samples to disk.

🔬 Processing Kepler-664...
  → Extracted and saved 46,042 samples to disk.

🔬 Processing Kepler-1123...
  → Extracted and saved 63,326 samples to disk.

🔬 Processing Kepler-1098...
  → Extracted and saved 56,093 samples to disk.

🔬 Processing HAT-P-49...
  → Extracted and saved 68,069 samples to disk.

🔬 Processing XO-4...
  → Extracted and saved 58,122 samples to disk.


STEP 3: Combining Data into Final HDF5 Dataset

Pass 1: Calculating total dataset size...
Total samples to combine: 3,501,577
Number of features: 7

Pass 2: Creating final HDF5 dataset at /content/drive/MyDrive/NASA_SpaceApps_2025/preprocessed_transit_data_FOLDED_v1.h5
  - Combined Kepler-334.npz (64,008 samples)
  - Combined Kepler-251.npz (36,479 samples)
  - Combined Kepler-729.npz (46,097 samples)
  - Combined Kepler-687.npz (34,487 samples)
  - Combined Kepler-119.npz (74,561 samples)
  - Combined Kepler-1115.npz (129,801 samples)
  - Combined Kepler-1073.npz (73,265 samples)
  - Combined Kepler-405.npz (16,748 samples)
  - Combined Kepler-1255.npz (36,408 samples)
  - Combined KELT-23_A.npz (273,305 samples)
  - Combined Kepler-1447.npz (130,086 samples)
  - Combined TOI-220.npz (549,879 samples)
  - Combined HD_23472.npz (207,312 samples)
  - Combined TOI-5726.npz (77,657 samples)
  - Combined TOI-2285.npz (151,946 samples)
  - Combined Kepler-659.npz (45,177 samples)
  - Combined HD_22946.npz (61,999 samples)
  - Combined Kepler-1118.npz (34,542 samples)
  - Combined Kepler-133.npz (125,156 samples)
  - Combined Kepler-620.npz (81,594 samples)
  - Combined Kepler-92.npz (163,600 samples)
  - Combined HAT-P-51.npz (32,719 samples)
  - Combined Kepler-179.npz (103,581 samples)
  - Combined Kepler-166.npz (67,300 samples)
  - Combined Kepler-595.npz (79,579 samples)
  - Combined Kepler-28.npz (72,630 samples)
  - Combined Kepler-210.npz (147,859 samples)
  - Combined Kepler-782.npz (79,002 samples)
  - Combined Kepler-414.npz (143,642 samples)
  - Combined Kepler-238.npz (69,506 samples)
  - Combined Kepler-664.npz (46,042 samples)
  - Combined Kepler-1123.npz (63,326 samples)
  - Combined Kepler-1098.npz (56,093 samples)
  - Combined HAT-P-49.npz (68,069 samples)
  - Combined XO-4.npz (58,122 samples)

✅ New accurately-labeled dataset created successfully!
  
</details>



# STEP 2: Creating Leak-Proof Train & Hold-out Test Sets

Training/Validation stars (28): Kepler-133, Kepler-1118, Kepler-210, Kepler-1123, TOI-5726, HAT-P-49, KELT-23 A, Kepler-238, Kepler-687, Kepler-1073, Kepler-1255, HAT-P-51, HD 22946, TOI-2285, HD 23472, Kepler-92, Kepler-595, TOI-220, Kepler-119, Kepler-414, Kepler-251, Kepler-659, Kepler-1098, Kepler-334, XO-4, Kepler-1115, Kepler-166, Kepler-729
Hold-out Test stars (7): Kepler-664, Kepler-1447, Kepler-620, Kepler-405, Kepler-179, Kepler-782, Kepler-28

Loading training data into RAM...
Loading hold-out test data into RAM...

Train set size: 2,971,894
Hold-out Test set size: 529,683


STEP 3: Training and Evaluating XGBoost Classifier

# Training model...
→ Model training complete.

# Evaluating on the UNSEEN Hold-out Test Set...

Hold-out Test Set Performance:
              precision    recall  f1-score   support

      Normal       0.98      0.65      0.79    521444
     Transit       0.01      0.30      0.03      8239

    accuracy                           0.65    529683
   macro avg       0.50      0.48      0.41    529683
weighted avg       0.97      0.65      0.77    529683

AUC-ROC on Hold-out Test Set: 0.4773

Confusion Matrix:


Precision-Recall Curve:
Area Under PR Curve (AUPRC): 0.0151



# STEP 4: Saving Production Model


→ Production model saved: /content/drive/MyDrive/NASA_SpaceApps_2025/exoplanet_detector_FOLDED_v1.pkl

TRAINING PIPELINE COMPLETE.


<img width="656" height="530" alt="download" src="https://github.com/user-attachments/assets/03c094c6-e4e0-4c76-b444-dd8bfa53768e" />




<img width="691" height="549" alt="download" src="https://github.com/user-attachments/assets/2016f525-fb4e-458a-95e0-1b30da0127e1" />

