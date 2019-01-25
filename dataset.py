import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
file = open("recognizer/notes.txt", "r")
data = file.readlines()[-1]
count = data.split(".")
iduser = int(count[0]) + 1
file.close()

video_capture = cv2.VideoCapture(0)
namauser = raw_input("Masukkan Nama Anda: ")
number = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        number = number + 1;
        cv2.imwrite("dataset/User."+str(iduser)+"."+str(number)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.waitKey(100);

    # Display the resulting frame
    cv2.imshow('Video', frame)

    cv2.waitKey(1);
    if (number >= 20):
        f = open("recognizer/notes.txt", "a")
        f.write("\n"+str(iduser)+"."+str(namauser))
        f.close()
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
