import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_and_preprocess_image
from image_model import build_image_only_model

# Directories
benign_dir = 'dataset/benign'
malignant_dir = 'dataset/malignant'

# Load only RGB image data
img_data, labels = [], []
print("[INFO] Loading RGB images only...")

for label, folder in enumerate([benign_dir, malignant_dir]):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            img, _, _ = load_and_preprocess_image(file_path)
            img_data.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Failed to load {file_path} – {e}")
            continue

img_data = np.array(img_data)
labels = to_categorical(np.array(labels))

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    img_data, labels, test_size=0.2, random_state=42
)

# Build model
model = build_image_only_model()

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights = dict(enumerate(class_weights))

# Train model
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=8,
    epochs=25,
    class_weight=class_weights,
    verbose=1
)

# Evaluate model
print("[INFO] Evaluating...")
loss, acc = model.evaluate(x_test, y_test)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# Classification Report & Confusion Matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=["Benign", "Malignant"]))

print("🧩 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model
model.save("image_only_model.h5")
print("[INFO] Saved 'image_only_model.h5'")