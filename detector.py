import os
import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
import Image

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
rec = cv2.face.LBPHFaceRecognizer_create();
path = "dataset"

def getImages(p):
    imgPaths = [os.path.join(path, file) for file in os.listdir(p)]
    pic = []
    ids = []
    for imgPath in imgPaths:
        temp = Image.open(imgPath).convert("L");
        face = np.array(temp, "uint8")
        idimg = int(os.path.split(imgPath)[-1].split(".")[1])
        pic.append(face)
        ids.append(idimg)
        #cv2.imshow("training", face)
        cv2.waitKey(10)
    return ids, pic

ids, pic = getImages(path)
rec.train(pic, np.array(ids))
rec.write("recognizer/training.yml")
cv2.destroyAllWindows()

rec.read("recognizer/training.yml")
iduser = 0
status = False
datas = []
dataid = []
datanama = []
file = open("recognizer/notes.txt", "r")
for line in file:
    datas.append(line)
for data in datas:
    temp = data.split(".")
    dataid.append(temp[0])
    nama = temp[1].split("\n")
    datanama.append(nama[0])
file.close()

video_capture = cv2.VideoCapture(0)

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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        iduser,conf = rec.predict(gray[y:y+h, x:x+w])
        for i in range(0, len(dataid)):
            if dataid[i] == str(iduser):
                cv2.putText(frame, str(datanama[i]), (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 4, 1);
                status = True
                break

        if status == False:
            cv2.putText(frame, "Unknown", (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 4, 1);
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
