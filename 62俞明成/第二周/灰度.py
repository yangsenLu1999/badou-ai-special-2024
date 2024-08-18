import cv2 as cv
import numpy as np


img = cv.imread('lenna.png')
h, w = img.shape[:2]

gray1 = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        temp = img[i, j]
        gray1[i, j] = int(temp[0] * 0.11 + temp[1] * 0.59 + temp[2] * 0.3)
cv.imshow("gray1", gray1)
cv.waitKey(0)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("gray", gray)
# cv.waitKey(0)

# 2值
bg = cv.imread("lenna.png", cv.IMREAD_GRAYSCALE)
print(bg)
img_binary = np.where(bg >= 127, 255, 0)
print(img_binary)
cv.imshow("binary", img_binary.astype(np.uint8))
cv.waitKey(0)









