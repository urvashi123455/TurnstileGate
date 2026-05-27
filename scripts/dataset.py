import cv2
import os

cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

user_id = input("Enter ID (1,2,3...): ")

count = 0

if not os.path.exists("dataset"):
    os.makedirs("dataset")

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1

        cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg",
                    gray[y:y+h, x:x+w])

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow('Face Capture', img)

    if count >= 150:
        break

cam.release()
cv2.destroyAllWindows()
