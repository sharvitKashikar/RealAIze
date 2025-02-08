import cv2
import dlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from imutils import face_utils
import os
import glob
import pickle  # 🔥 Added for saving feature_length

# Load Dlib's face detector and 68-landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Paths to dataset
dataset_path = "/Users/sharvitkashikar/Downloads/SMR/Dataset 2/Train"
real_path = os.path.join(dataset_path, "Real")
fake_path = os.path.join(dataset_path, "Fake")

# Extract facial landmarks
def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        return shape  # Return landmarks as an array

# Convert landmarks into feature vectors
def extract_features(landmarks):
    if landmarks is None:
        return None
    return np.array(landmarks).flatten()  # Convert (68,2) to (136,)

# Detect color artifacts
def check_color_artifacts(image_path):
    image = cv2.imread(image_path)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    std_dev = np.std(ycrcb[:, :, 1:3])
    return std_dev  # Higher values indicate artifacts

# Check for JPEG compression artifacts
def jpeg_artifact_score(image_path):
    image = cv2.imread(image_path)
    edges = cv2.Canny(image, 100, 200)
    return np.mean(edges)

# Train the CNN Model
def train_cnn_model():
    print("🚀 Training Deepfake Detector...")

    # Load dataset paths
    real_images = glob.glob(os.path.join(real_path, "*.jpg"))[:500]  
    fake_images = glob.glob(os.path.join(fake_path, "*.jpg"))[:500]  

    print(f"✅ Found {len(real_images)} Real Images and {len(fake_images)} Fake Images")

    X, y = [], []

    # Extract features from real images
    for img_path in real_images:
        landmarks = extract_landmarks(img_path)
        if landmarks is not None:
            features = extract_features(landmarks)
            features = np.append(features, check_color_artifacts(img_path))
            features = np.append(features, jpeg_artifact_score(img_path))
            X.append(features)
            y.append(0)  # 0 = Authentic

    # Extract features from fake images
    for img_path in fake_images:
        landmarks = extract_landmarks(img_path)
        if landmarks is not None:
            features = extract_features(landmarks)
            features = np.append(features, check_color_artifacts(img_path))
            features = np.append(features, jpeg_artifact_score(img_path))
            X.append(features)
            y.append(1)  # 1 = Deepfake

    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    print("✅ Dataset loaded successfully!")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Debug: Print dataset shapes
    feature_length = X_train.shape[1]  # Auto-detect feature length
    print("Feature Length:", feature_length)

    X_train = X_train.reshape(X_train.shape[0], feature_length, 1, 1)  
    X_test = X_test.reshape(X_test.shape[0], feature_length, 1, 1)  

    print("X_train shape after reshape:", X_train.shape)

    # ✅ FIX: Reduce Kernel Size to Avoid Negative Shape
    model = Sequential([
        Input(shape=(feature_length, 1, 1)),  
        Conv2D(32, (1,1), activation='relu'),  
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification (Real vs Deepfake)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the model
    model.save("deepfake_detector.h5")
    print("✅ CNN Model Trained and Saved!")

    # 🔥 FIX: Save feature_length
    with open("feature_length.pkl", "wb") as file:
        pickle.dump(feature_length, file)
    print("✅ Feature length saved!")

# Detect deepfake using the trained model
def detect_deepfake(image_path):
    # Load trained CNN model
    if not os.path.exists("deepfake_detector.h5"):
        print("⚠️ Model not found! Train it first using train_cnn_model().")
        return

    model = tf.keras.models.load_model("deepfake_detector.h5")

    # 🔥 FIX: Load feature_length from file
    feature_length_file = "feature_length.pkl"
    
    if not os.path.exists(feature_length_file):
        print("⚠️ Feature length file missing. Retraining model...")
        train_cnn_model()

    with open(feature_length_file, "rb") as file:
        feature_length = pickle.load(file)

    landmarks = extract_landmarks(image_path)
    if landmarks is None:
        return "❌ No Face Detected."

    # Extract features
    features = extract_features(landmarks)
    features = np.append(features, check_color_artifacts(image_path))
    features = np.append(features, jpeg_artifact_score(image_path))
    
    # ✅ FIX: Trim or Pad Feature Vector to match expected `feature_length`
    if len(features) > feature_length:
        features = features[:feature_length]  # Trim excess
    elif len(features) < feature_length:
        features = np.pad(features, (0, feature_length - len(features)))  # Pad if too short

    features = np.array(features).reshape(1, feature_length, 1, 1)  # ✅ FIX: Use Correct Shape

    # Prediction
    prediction = model.predict(features)
    return "⚠️ Deepfake Detected!" if prediction >= 0.5 else "✅ Authentic Image."

# Train the CNN model if not already trained
if not os.path.exists("deepfake_detector.h5") or not os.path.exists("feature_length.pkl"):
    train_cnn_model()

# Test with an image
test_image = "/Users/sharvitkashikar/Downloads/SMR/Images/smr.JPG"
print("\n🔍 Running Deepfake Detection on:", test_image)
print(detect_deepfake(test_image))