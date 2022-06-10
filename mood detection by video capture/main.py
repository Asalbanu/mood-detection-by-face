import cv2
from deepface import DeepFace
import numpy as np

face_cascade_name = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print("Error loading xml file")

video = cv2.VideoCapture(0)
while True:
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame', gray)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    #
    for (x, y, w, h) in face:
        cv2.rectangle(frame,(x, y), (x + w, y + h), (0, 0, 255), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        analyze = DeepFace.analyze(frame, actions=['emotion'])
        cv2.putText(frame,
                    analyze['dominant_emotion'],
                    (50,50),
                    font, 3,
                    (0,255,255),
                    2,
                    cv2.LINE_4)
    except:
        cv2.putText(frame,
                    "No face found!",
                    (50, 50),
                    font, 3,
                    (0, 255, 255),
                    2,
                    cv2.LINE_4)
    cv2.imshow('video', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyALlWindows()