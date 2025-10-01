
# Data Requrments


```
"2-minute cadence data" means that the TESS satellite took a picture and measured the brightness of that specific star once every two minutes.
Let's break that down with an analogy and explain why it's so important.
The Flipbook Analogy
Imagine you are trying to capture a video of a fast-moving bird flying across the sky, but you can only take still pictures.
Long Cadence (e.g., 30 minutes): If you take one picture every 30 minutes, you might get one blurry image of the bird at the beginning, one in the middle, and one at the end. You'll know a bird was there, but you won't see how it flew, its path, or its shape clearly. You might even miss it entirely if it flew by between your pictures.
Short Cadence (e.g., 2 minutes): If you take one picture every two minutes, you will get a smooth series of images. You can stack them together to create a flipbook and see the bird's exact path, its speed, and its shape in great detail.

In our project:
The "bird" is the exoplanet transit (the dip in brightness).
The "pictures" are the brightness measurements taken by TESS.
The "cadence" is the time between each measurement.
Why 2-Minute Cadence is Crucial for Your AI Model
Capturing the Transit Shape: A typical exoplanet transit lasts for a few hours.
With 2-minute cadence, a 3-hour transit (180 minutes) is captured with 90 data points (180 / 2). This gives a clear, well-defined "U" shape that the AI can easily learn to recognize.
With TESS's older, lower-resolution data (30-minute cadence), the same 3-hour transit would only have 6 data points (180 / 30). This is not enough information to define the shape, and the AI would struggle to distinguish it from random noise.
Higher Data Quality: The 2-minute data is specifically designated for pre-selected "Targets of Interest" that scientists want to study closely. This data is often of higher quality and has been processed more carefully, making it ideal for training a machine learning model.
The Purpose of the Script's Check: Your new, improved script first queries a massive catalog of all known exoplanets. However, not all of them were targeted for these special, high-resolution observations. The line Checking for available 2-minute cadence data... is a critical filtering step where the code checks, "For this list of interesting stars, which ones actually have the high-quality, 2-minute 'flipbook' data we need to train our AI?"
If a star only has the low-resolution (long cadence) data, your script correctly discards it because it's not suitable for building a reliable detection model.
In summary, "2-minute cadence" refers to high-frequency, high-resolution brightness measurements, which are essential for capturing the detailed shape of an exoplanet transit and successfully training your AI model.
22.3s
Use Arrow Up and Arrow Down to select a turn, Enter to jump to it, and Escape to return to the chat.
```
