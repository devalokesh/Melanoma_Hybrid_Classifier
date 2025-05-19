import numpy as np
import cv2
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess_image
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load trained model
model = load_model("hybrid_tumor_model.h5")

# Open a file dialog to select image
Tk().withdraw()  # Hide the root window
img_path = askopenfilename(title="Select an image to classify", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

if not img_path:
    print("No file selected.")
    exit()

# Preprocess the image
img_rgb, edge_img, hist_vec = load_and_preprocess_image(img_path)

# Prepare data shape
img_rgb = np.expand_dims(img_rgb, axis=0)
edge_img = np.expand_dims(edge_img, axis=0)
hist_vec = np.expand_dims(hist_vec, axis=0)

# Run prediction
prediction = model.predict([img_rgb, edge_img, hist_vec])
label = "Malignant" if np.argmax(prediction) == 1 else "Benign"

print(f"\n🧬 Prediction Result: {label}")
