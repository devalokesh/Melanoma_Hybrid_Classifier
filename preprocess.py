# preprocess.py

import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_image(path):
    """
    Reads an image, resizes it to (50, 50), and prepares:
    - RGB image (3 channels)
    - Canny edge map (1 channel)
    - Histogram (256-bin grayscale)
    """

    # Load and resize
    img = cv2.imread(path)
    img = cv2.resize(img, (50, 50))

    # Normalize RGB image to 0-1
    img_rgb = img.astype(np.float32) / 255.0

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection (Canny)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    edges = edges.astype(np.float32) / 255.0
    edges = np.expand_dims(edges, axis=-1)  # shape: (50, 50, 1)

    # Histogram (grayscale, 256 bins)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # Normalize histogram to 0-1
    scaler = MinMaxScaler()
    hist = scaler.fit_transform(hist.reshape(-1, 1)).flatten()

    return img_rgb, edges, hist
