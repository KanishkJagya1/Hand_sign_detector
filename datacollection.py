import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

detector = HandDetector(maxHands=1)
cap = cv2.VideoCapture(0)

offset = 20
imageSize = 300

folder = "./python/data/C"
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

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        counter += 1
        filename = f"{folder}/Image_{time.time()}.jpg"
        cv2.imwrite(filename, imgWhite)
        print(f"Image saved as {filename}, total images: {counter}")

cap.release()
cv2.destroyAllWindows()
