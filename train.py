
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from preprocess import load_and_preprocess_image
from hybrid_model import build_hybrid_model
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping


benign_dir = 'DataSet/benign'
malignant_dir = 'DataSet/malignant'

img_data, edge_data, hist_data, labels = [], [], [], []

print("[INFO] Loading and preprocessing images...")

# Load benign images
for file in os.listdir(benign_dir):
    try:
        img_path = os.path.join(benign_dir, file)
        img, edge, hist = load_and_preprocess_image(img_path)
        img_data.append(img)
        edge_data.append(edge)
        hist_data.append(hist)
        labels.append(0)
    except:
        print(f"[WARN] Skipped {file} (benign)")

# Load malignant images
for file in os.listdir(malignant_dir):
    try:
        img_path = os.path.join(malignant_dir, file)
        img, edge, hist = load_and_preprocess_image(img_path)
        img_data.append(img)
        edge_data.append(edge)
        hist_data.append(hist)
        labels.append(1)
    except:
        print(f"[WARN] Skipped {file} (malignant)")

# 🧪 Check class balance
print("[INFO] Class distribution:", Counter(labels))

# Balance dataset (optional: undersample or oversample)
# Here, we undersample the majority class
from imblearn.under_sampling import RandomUnderSampler

X = list(zip(img_data, edge_data, hist_data))
y = labels

shapes = [x[0].shape for x in X]
if not all(s == shapes[0] for s in shapes):
    raise ValueError("[ERROR] Inconsistent image shapes in dataset.")


# Flatten for sampling
flat_X = np.arange(len(X)).reshape(-1, 1)
rus = RandomUnderSampler()
flat_X_res, y_res = rus.fit_resample(flat_X, y)

# Build balanced dataset
X_resampled = [X[i[0]] for i in flat_X_res]
img_data, edge_data, hist_data = zip(*X_resampled)
labels = y_res

print("[INFO] New class distribution:", Counter(labels))

# Convert to arrays
img_data = np.array(img_data)
edge_data = np.array(edge_data)
hist_data = np.array(hist_data)
labels = to_categorical(np.array(labels))

# Split
x_train_img, x_test_img, x_train_edge, x_test_edge, x_train_hist, x_test_hist, y_train, y_test = train_test_split(
    img_data, edge_data, hist_data, labels, test_size=0.2, random_state=42, shuffle=True
)

# Build and train model
model = build_hybrid_model()

print("[INFO] Training model...")

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


model.fit(
    [x_train_img, x_train_edge, x_train_hist], y_train,
    validation_data=([x_test_img, x_test_edge, x_test_hist], y_test),
    epochs=30, batch_size=8, shuffle=True, verbose=1
)

# Save model
model.save("hybrid_tumor_model.h5")
print("[INFO] Model saved as 'hybrid_tumor_model.h5'")
