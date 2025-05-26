import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_and_preprocess_image
from image_model import build_image_only_model

# Define train and test directories
train_dir = 'train'
test_dir = 'test'

def load_dataset(base_dir):
    img_data, labels = [], []
    for label, folder in enumerate(['benign', 'malignant']):
        folder_path = os.path.join(base_dir, folder)
        for file in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, file)
                img, _, _ = load_and_preprocess_image(img_path)
                img_data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
    return np.array(img_data), to_categorical(np.array(labels))

# Load data
print("[INFO] Loading training data...")
x_train, y_train = load_dataset(train_dir)
print("[INFO] Loading test data...")
x_test, y_test = load_dataset(test_dir)

# Class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights = dict(enumerate(class_weights))

# Build model
model = build_image_only_model()

# Train
print("[INFO] Training model...")
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=8,
    epochs=25,
    class_weight=class_weights,
    verbose=1
)

# Evaluate
print("[INFO] Evaluating...")
loss, acc = model.evaluate(x_test, y_test)
print(f"✅ Test Accuracy: {acc*100:.2f}%")
print(f"📉 Test Loss: {loss:.4f}")

# Predict
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=["Benign", "Malignant"]))

# Confusion Matrix
print("🧩 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Accuracy Plot
plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid()
plt.show()

# Save
model.save("train_test_image_model.h5")
print("[INFO] Model saved as 'train_test_image_model.h5'")
