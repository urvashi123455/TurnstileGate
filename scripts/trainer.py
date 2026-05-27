import cv2
import numpy as np
from PIL import Image
import os

# Path of dataset folder
path = 'dataset'

# Create LBPH Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Haarcascade file
detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ---------------- FUNCTION ----------------
def get_images_and_labels(path):

    face_samples = []
    ids = []

    # Get image paths
    image_paths = [
        os.path.join(path, f)
        for f in os.listdir(path)
    ]

    for image_path in image_paths:

        try:
            # Convert image to grayscale
            pil_img = Image.open(image_path).convert('L')

            # Convert image into numpy array
            img_numpy = np.array(pil_img, 'uint8')

            # Get filename
            file_name = os.path.split(image_path)[-1]

            # Split filename
            parts = file_name.split(".")

            # Skip wrong files
            if len(parts) < 4:
                print(f"Skipped invalid file: {file_name}")
                continue

            # Get ID
            id = int(parts[1])

            # Detect face
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:

                face_samples.append(
                    img_numpy[y:y+h, x:x+w]
                )

                ids.append(id)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return face_samples, ids

# ---------------- TRAIN MODEL ----------------
print("\n[INFO] Training faces. Please wait...")

faces, ids = get_images_and_labels(path)

# Train recognizer
recognizer.train(faces, np.array(ids))

# Save trained model
recognizer.write('trainer.yml')

print("\n[INFO] Model trained successfully!")
print(f"[INFO] Total faces trained: {len(np.unique(ids))}")
