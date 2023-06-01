import face_recognition
import cv2
import dlib
import numpy as np
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Preprocessing: Face Alignment
def align_face(image):

    # Preprocessing: Face Alignment using Dlib
    def align_face(image):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat.bz2")

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = detector(gray)

        aligned_faces = []
        for face in faces:
            # Predict facial landmarks
            landmarks = predictor(gray, face)

            # Perform face alignment using the detected landmarks
            aligned_face = dlib.get_face_chip(image, landmarks)
            aligned_faces.append(aligned_face)

        return aligned_faces

# Preprocessing: Illumination Normalization
def normalize_illumination(image):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l)

    # Merge the enhanced L channel with the original A and B channels
    enhanced_lab = cv2.merge((clahe_l, a, b))

    # Convert the LAB image back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_image
# Preprocessing: Face Detection and Cropping
def detect_and_crop_faces(image):
    # Initialize the MTCNN face detector
    detector = MTCNN()

    # Perform face detection to locate faces in the image using MTCNN
    faces = detector.detect_faces(image)

    # Crop the detected faces from the image
    cropped_faces = []
    for face in faces:
        x, y, w, h = face['box']
        face_image = image[y:y + h, x:x + w]
        cropped_faces.append(face_image)

    return cropped_faces


# Splitting the Dataset
def split_dataset(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Performing Face Recognition
def perform_face_recognition(X_probe, y_probe, X_gallery, y_gallery):
    y_pred = []
    for i in range(len(X_probe)):
        probe_encoding = face_recognition.face_encodings(X_probe[i])[0]
        distances = face_recognition.face_distance(X_gallery, probe_encoding)
        min_distance_index = np.argmin(distances)
        if distances[min_distance_index] < matching_threshold:
            y_pred.append(y_gallery[min_distance_index])
        else:
            y_pred.append("Unknown")
    return y_pred


# Defining the Matching Threshold
matching_threshold = 0.6

# Evaluating Performance Metrics
def evaluate_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, precision, recall, f1

# Example Usage
# Replace [...] with your actual dataset and labels
X = [...]  # Dataset of images
y = [...]  # Labels corresponding to the images

# Preprocessing Step: Detect, Align, and Crop Faces
X_processed = []
for image in X:
    # Preprocessing: Face Alignment
    aligned_image = align_face(image)
    # Preprocessing: Illumination Normalization
    normalized_image = normalize_illumination(aligned_image)
    # Preprocessing: Face Detection and Cropping
    cropped_faces = detect_and_crop_faces(normalized_image)

    X_processed.extend(cropped_faces)

X_train, X_test, y_train, y_test = split_dataset(X_processed, y)

# Train and Test split
# Perform any necessary feature extraction or normalization on X_train and X_test

# Perform Face Recognition on the Probe Set
y_pred = perform_face_recognition(X_test, y_test, X_train)

# Evaluate Performance Metrics
accuracy, precision, recall, f1 = evaluate_performance(y_test, y_pred)

# Print the performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
