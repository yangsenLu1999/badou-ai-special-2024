# -*- coding: utf-8 -*-
'''
@ Author: Hayne
@ Time: 2024/03/28
'''

import cv2
import numpy as np


def nearest_inter(img, zoom_h, zoom_w):
    height, width, channels = img.shape
    inter_image = np.zeros((zoom_h, zoom_w, channels), np.uint8)
    sh = height / zoom_h
    sw = width / zoom_w
    for i in range(zoom_h):
        for j in range(zoom_w):
            x = int(i * sh + 0.5)
            y = int(j * sw + 0.5)
            inter_image[i, j] = img[x, y]
    return inter_image


# cv2.resize(img, (800,800,c),near/bin)

img = cv2.imread("lenna.png")
zoom = nearest_inter(img, 1000, 1000)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
