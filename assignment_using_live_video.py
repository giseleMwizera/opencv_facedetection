import cv2
import os

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier('C:/Users/Bijoux/Desktop/DATA/Lessons/YR3/ADVANCED_EMBEDDED/third_term/opencv/haarcascade_frontalface_default.xml')

# Start the video stream from the default camera
cap = cv2.VideoCapture(0)

# Create a directory to save the cropped faces
if not os.path.exists('cropped_faces'):
    os.makedirs('cropped_faces')

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Crop and save all detected faces
    for i, (x, y, w, h) in enumerate(faces):
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite(f'C:/Users/Bijoux/Desktop/DATA/Lessons/YR3/ADVANCED_EMBEDDED/third_term/opencv/cropped_faces/face_{i}.jpg', face_img)

    # Display the original frame with rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video Stream', frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
