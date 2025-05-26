import os
import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess_image

# Paths
dataset_dir = 'train'  # or 'dataset', 'data' etc. depending on structure

def summarize_features(folder):
    features = []
    print(f"📂 Processing folder: {folder}")
    for label, category in enumerate(['benign', 'malignant']):
        path = os.path.join(folder, category)
        for file in os.listdir(path):
            try:
                full_path = os.path.join(path, file)
                img, _, _ = load_and_preprocess_image(full_path)
                mean_val = np.mean(img)
                std_val = np.std(img)
                features.append((label, mean_val, std_val))
            except Exception as e:
                print(f"Failed on {file}: {e}")
    return np.array(features)

features = summarize_features(dataset_dir)

benign_means = features[features[:, 0] == 0][:, 1].astype(float)
malignant_means = features[features[:, 0] == 1][:, 1].astype(float)

plt.hist(benign_means, bins=30, alpha=0.6, label='Benign')
plt.hist(malignant_means, bins=30, alpha=0.6, label='Malignant')
plt.title("Histogram of Mean Pixel Intensities")
plt.xlabel("Mean Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show()