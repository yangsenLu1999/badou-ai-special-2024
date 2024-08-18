import cv2 as cv
import numpy as np

img = cv.imread('lenna.png')
h, w, c = img.shape
Image = np.zeros((800, 800, c), np.uint8)
Xx = 800 / h
Yy = 800 / w
for i in range(800):
    for j in range(800):
        x = int(i / Xx + 0.5)
        y = int(j / Yy + 0.5)
        Image[i, j] = img[x, y]
cv.imshow("Image", Image)
cv.waitKey(0)
