import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import os

detector = HandDetector(maxHands=1)
cap = cv2.VideoCapture(0)

classifier = Classifier("./python/model/keras_model.h5", "./python/model/labels.txt")
offset = 20
imageSize = 300

folder = "./python/data/C"  # Provide the full path to your data folder
counter = 0

if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Resize imgCrop to match imgWhite's dimensions
        imgCrop = cv2.resize(imgCrop, (imageSize, imageSize))

        aspect_ratio = w / h  # Calculate aspect ratio

        if aspect_ratio > 1:
            # If the width is greater than the height
            k = imageSize / w
            w_cal = imageSize
            h_cal = int(h * k)
        else:
            # If the height is greater than the width
            k = imageSize / h
            w_cal = int(w * k)
            h_cal = imageSize

        imgResize = cv2.resize(imgCrop, (w_cal, h_cal))  # Resize the cropped image
        h_gap = int((imageSize - h_cal) / 2)
        w_gap = int((imageSize - w_cal) / 2)

        imgWhite[h_gap:h_gap + h_cal, w_gap:w_gap + w_cal] = imgResize

        # Check if the hand is in a closed fist gesture (modify based on hand landmarks)
        if detector.isFist(hand):
            text = "3 C"
        else:
            text = ""

        cv2.putText(img, text, (x - offset, y - offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
