import cv2
import dlib
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from imutils import face_utils

# ✅ Load OpenCV's Haarcascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ✅ Load Dlib's face detector & 68-landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# ✅ Paths to dataset
dataset_path = "/Users/sharvitkashikar/Downloads/SMR/Dataset 2/Train"
real_path = os.path.join(dataset_path, "Real")
fake_path = os.path.join(dataset_path, "Fake")

# ✅ Convert landmarks into an image
def landmarks_to_image(landmarks):
    blank_image = np.zeros((224, 224, 3), dtype=np.uint8)
    for (x, y) in landmarks:
        cv2.circle(blank_image, (x, y), 2, (255, 255, 255), -1)
    return blank_image

# ✅ Extract facial landmarks and return as an image
def extract_landmark_image(image_path):
    print(f"🔍 Loading Image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ ERROR: Cannot read image {image_path}")
        return None

    # ✅ Display image for debugging
    cv2.imshow("Loaded Image", image)
    cv2.waitKey(1000)  # Show for 1 second
    cv2.destroyAllWindows()

    # ✅ Resize image before detection
    image = cv2.resize(image, (400, 400))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ✅ Try OpenCV face detector first
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print(f"❌ No Face Detected in {image_path} using OpenCV. Trying Dlib...")
        faces = detector(gray)

    if len(faces) == 0:
        print("❌ No Face Detected with Dlib as well!")
        return None

    # ✅ Extract first detected face
    if isinstance(faces, np.ndarray):  # OpenCV returns an array
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
    else:  # Dlib returns a rectangle object
        x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
        face_roi = gray[y:y+h, x:x+w]

    # ✅ Resize extracted face to ResNet50 format
    face_roi = cv2.resize(face_roi, (224, 224))

    # ✅ Display extracted face for debugging
    cv2.imshow("Detected Face", face_roi)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    return face_roi  # Return detected face image

# ✅ Train ResNet50 Model
def create_resnet_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False  # Freeze layers

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

    return model

# ✅ Train Model
def train_resnet_model():
    print("🚀 Training ResNet50 Deepfake Detector...")

    # ✅ Load dataset
    real_images = glob.glob(os.path.join(real_path, "*.jpg"))[:2000]
    fake_images = glob.glob(os.path.join(fake_path, "*.jpg"))[:2000]

    X, y = [], []

    # ✅ Extract features from real images
    for img_path in real_images:
        landmark_img = extract_landmark_image(img_path)
        if landmark_img is not None:
            X.append(landmark_img)
            y.append(0)

    # ✅ Extract features from fake images
    for img_path in fake_images:
        landmark_img = extract_landmark_image(img_path)
        if landmark_img is not None:
            X.append(landmark_img)
            y.append(1)

    # ✅ Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # ✅ Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Train ResNet50
    model = create_resnet_model()
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # ✅ Save model
    model.save("deepfake_detector_resnet.h5")
    print("✅ Model trained & saved!")

# ✅ Detect Deepfake
def detect_deepfake(image_path):
    if not os.path.exists("deepfake_detector_resnet.h5"):
        train_resnet_model()

    model = load_model("deepfake_detector_resnet.h5")

    landmark_img = extract_landmark_image(image_path)
    if landmark_img is None:
        return "❌ No Face Detected."

    # ✅ Ensure correct input shape
    landmark_img = np.expand_dims(landmark_img, axis=0)

    prediction = model.predict(landmark_img)[0][0]

    return "⚠️ Deepfake Detected!" if prediction >= 0.5 else "✅ Authentic Image."

# ✅ Train if needed
if not os.path.exists("deepfake_detector_resnet.h5"):
    train_resnet_model()

# ✅ Test Image
test_image = "/Users/sharvitkashikar/Downloads/SMR/Images/mru.JPG"
print("\n🔍 Running Deepfake Detection on:", test_image)
print(detect_deepfake(test_image))
