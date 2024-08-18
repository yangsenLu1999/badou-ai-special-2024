# -*- coding: utf-8 -*-
'''
@ Author: Hayne
@ Time: 2024/03/28
'''

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread(r"lenna.png")
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)

# gray
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
# img_gray = rgb2gray(img)
print(img_gray)

# binary
img_bin = np.zeros([h,w], img.dtype)
for p in range(h):
    for q in range(w):
        if img_gray[p, q]/255 <= 0.5:
            img_bin[p, q] = 0
        else:
            img_bin[p, q] = 1
# img_bin = np.where(img_gray >= 0.5, 1, 0)


plt.subplot(221)
img = plt.imread(r"lenna.png")
plt.imshow(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

plt.subplot(223)
plt.imshow(img_bin, cmap='gray')
plt.show()
