import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("train_test_image_model.h5")  # Make sure this file exists in your folder

def predict_frame(image):
    # Resize to match model input size
    img = cv2.resize(image, (50, 50))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)
    label = "Malignant" if np.argmax(pred) == 1 else "Benign"
    confidence = np.max(pred) * 100
    return label, confidence

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not access the webcam.")
    exit()

print("📷 Press SPACEBAR to capture and classify the image.")
print("🛑 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show live feed
    cv2.imshow("Tumor Classifier - Press SPACE to Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE pressed
        # Freeze frame and predict
        snapshot = frame.copy()
        label, confidence = predict_frame(snapshot)

        # Show result with overlay
        result_img = snapshot.copy()
        text = f"{label} ({confidence:.2f}%)"
        color = (0, 0, 255) if label == "Malignant" else (0, 255, 0)
        cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Prediction Result", result_img)

        print(f"🧬 Prediction: {label} ({confidence:.2f}%)")
        cv2.waitKey(0)  # Wait for any key before resuming
        cv2.destroyWindow("Prediction Result")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
