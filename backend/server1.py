from flask import Flask, request, jsonify, render_template
import cv2
import dlib
import numpy as np
import tensorflow as tf
import os
import pickle
from imutils import face_utils
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load Dlib's face detector and 68-landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load trained CNN model
MODEL_PATH = "deepfake_detector.h5"
FEATURE_LENGTH_PATH = "feature_length.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_LENGTH_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(FEATURE_LENGTH_PATH, "rb") as file:
        feature_length = pickle.load(file)
else:
    raise FileNotFoundError("Trained model or feature length file missing. Train the model first.")

# Extract facial landmarks
def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    for face in faces:
        shape = predictor(gray, face)
        return face_utils.shape_to_np(shape)

# Convert landmarks into feature vectors
def extract_features(landmarks):
    if landmarks is None:
        return None
    return np.array(landmarks).flatten()

# Detect color artifacts
def check_color_artifacts(image_path):
    image = cv2.imread(image_path)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return np.std(ycrcb[:, :, 1:3])

# Check for JPEG compression artifacts
def jpeg_artifact_score(image_path):
    image = cv2.imread(image_path)
    edges = cv2.Canny(image, 100, 200)
    return np.mean(edges)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/deepfake/detect", methods=["POST"])
def detect_deepfake():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    file.save(file_path)
    
    landmarks = extract_landmarks(file_path)
    if landmarks is None:
        return jsonify({"error": "No face detected in the image."}), 400
    
    features = extract_features(landmarks)
    features = np.append(features, check_color_artifacts(file_path))
    features = np.append(features, jpeg_artifact_score(file_path))
    
    if len(features) > feature_length:
        features = features[:feature_length]
    elif len(features) < feature_length:
        features = np.pad(features, (0, feature_length - len(features)))
    
    features = np.array(features).reshape(1, feature_length, 1, 1)
    prediction = model.predict(features)
    result = "Deepfake Detected!" if prediction >= 0.5 else "Authentic Image."
    
    return jsonify({"result": result})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, port=5001)
