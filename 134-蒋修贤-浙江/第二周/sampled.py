#!/usr/bin/env python3

import cv2
import numpy as np

#插值法放大图像
img = cv2.imread("68641863_p0.png")

dh = 2000
dw = 1416

h,w,c = img.shape
doubleImg = np.zeros((dh,dw,c),np.uint8)

sh = dh/h
sw = dw/w

for i in range(dh):
	for j in range(dw):
		x = int(i/sh-0.5)
		y = int(j/sw-0.5)
		doubleImg[i,j] = img[x,y]

cv2.imshow('img',img)
cv2.imshow('doubleImg',doubleImg)
cv2.waitKey(0)