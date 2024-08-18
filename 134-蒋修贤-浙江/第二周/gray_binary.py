#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread("68641863_p0.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#原图
plt.subplot(221)
plt.imshow(img)

#灰度化
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
	for j in range(w):
		m = img[i,j]
		img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

plt.subplot(222)
plt.imshow(cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB),cmap='gray')

#二值化
img_binary = img_gray
rows,cols = img_binary.shape
for i in range(rows):
	for j in range(cols):
		if(img_binary[i,j]<=192):
			img_binary[i,j] = 0
		else:
			img_binary[i,j] = 255

plt.subplot(223)
plt.imshow(cv2.cvtColor(img_binary, cv2.COLOR_BGR2RGB),cmap='gray')
plt.show()
