import os
import cv2
import numpy as np
import Image

rec = cv2.face.LBPHFaceRecognizer_create();
path = "dataset"

def getImages(p):
    imgPaths = [os.path.join(path, file) for file in os.listdir(p)]
    frame = []
    ids = []
    for imgPath in imgPaths:
        temp = Image.open(imgPath).convert("L");
        face = np.array(temp, "uint8")
        idimg = int(os.path.split(imgPath)[-1].split(".")[1])
        frame.append(face)
        ids.append(idimg)
        #cv2.imshow("training", face)
        cv2.waitKey(10)
    return ids, frame

ids, frame = getImages(path)
rec.train(frame, np.array(ids))
rec.write("recognizer/training.yml")
cv2.destroyAllWindows()