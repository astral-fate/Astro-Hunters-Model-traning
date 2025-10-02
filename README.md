# Astro-Hunters-Model-traning

- https://registry.opendata.aws/tess/

 
## The TESS Mission: A Planet-Hunting Satellite 🔭

**TESS**, which stands for **Transiting Exoplanet Survey Satellite**, is a NASA space telescope launched in 2018. Its single, specific mission is to discover thousands of exoplanets—planets orbiting stars other than our Sun.

TESS finds planets using the **transit method**. This method works by detecting a tiny, periodic dip in a star's brightness. This dip is caused by a planet passing in front of the star from our point of view, momentarily blocking a small fraction of its light, much like a moth flying in front of a distant streetlight. By measuring how much the light dims and how long the dimming lasts, scientists can determine the planet's size and orbital period. 

---
## TESS's Observation Strategy

### The 27-Day "Sector" System
To map the sky efficiently, TESS uses a clever strategy. It has four powerful cameras that observe a large, continuous strip of the sky called a **sector** for approximately **27 days**. After completing observations for one sector, the satellite repositions itself to observe the next adjacent sector. Over its primary mission, it mapped about 85% of the entire sky, creating a rich dataset for astronomers. This long observation period per sector is crucial for detecting planets with longer orbits.

### Data Downlink to Earth
TESS doesn't stream data live. It stores all the observations from a 27-day sector on board. Then, thanks to a unique high-earth orbit, it makes a close pass by Earth and beams the stored data down to the Deep Space Network. Once on the ground, the raw data goes through a processing pipeline at NASA to be calibrated and prepared for scientific analysis before being released to the public.

### The Data Product: Light Curves
The primary type of data TESS collects is **photometry**—the precise measurement of starlight brightness over time. When you plot this data, you get a **light curve**, which is a graph of brightness versus time. This is the fundamental dataset you are working with. TESS provides this data in two main forms:
* **Two-Minute Cadence:** For thousands of pre-selected target stars, TESS saves a brightness measurement every two minutes. This is the high-resolution data you are using, which is ideal for clearly defining short transit events.
* **Full-Frame Images (FFI):** TESS also saves a full image from all its cameras every 10-30 minutes. This allows scientists to create light curves for every object in its field of view, not just the pre-selected targets.

---
## Key Data Sources Explained

It's important to distinguish between the two main data sources you've used:

1.  **The TESS Data Archive (at MAST):** This is the official repository for the raw and processed data from the satellite itself, including all the light curves and full-frame images. You access this using tools like `lightkurve`.
2.  **The NASA Exoplanet Archive:** This is a **catalog** or encyclopedia. It contains a list of all *confirmed and validated* exoplanets discovered by TESS and other missions. You used this archive in your first script to get a list of "ground truth" stars known to have planets, which is perfect for creating a training dataset.

---
## Project's Goal and Method 🎯

Your project has two main phases, mirroring the process of real scientific discovery:

1.  **Phase 1: Supervised Training.** You first build a dataset of stars that the NASA Exoplanet Archive confirms have transiting planets. You process their light curves and use the `IsolationForest` algorithm to automatically label the transit dips. You then train an **XGBoost classification model** on this labeled data to teach it how to recognize the specific features of a real transit signal.

2.  **Phase 2: Prediction and Discovery.** In this phase, you use your trained model as a discovery tool. You download fresh TESS data for stars that are *not* in the archive (i.e., stars with no known planets). You apply the same processing pipeline and then use your model to predict whether any transit-like signals exist in these new light curves. The visualization script you created is the final step, allowing you to visually inspect the signals your model has flagged as potential new planet candidates.

---
## The Data Filtration Pipeline

To prepare the raw TESS data for your model, you perform a series of crucial filtering and cleaning steps. The new data for prediction **must** undergo the exact same steps as the training data.

* **Stitching (`.stitch()`):** Combines multiple observation sectors of the same star into one continuous light curve.
* **Removing Bad Data (`.remove_nans()`):** Deletes any data points that are "Not a Number" (NaNs), which represent errors or gaps in the observation.
* **Normalization:** You divide the entire light curve by its median brightness. This standardizes the data, centering the star's baseline brightness around a value of 1.0. This is essential for comparing different stars and for the model to work effectively.
* **Detrending (`median_filter`):** This is a critical step to remove "noise." Stars naturally vary in brightness over long periods due to stellar activity (like starspots) or slight instrumental drift. The median filter smooths out these slow, long-term trends, which helps to isolate the sharp, short-duration dips caused by a planetary transit. This makes the signal you're looking for much cleaner and more prominent.
