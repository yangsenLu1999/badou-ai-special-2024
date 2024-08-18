import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# 原图
plt.subplot(221)
img = plt.imread("c4b591762800e7b417922ee4bcfb4cd.jpg")
plt.imshow(img)

# 灰度图
plt.subplot(222)
h, w = img.shape[:2]
img2 = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]   # 取出原图像的每个点
        img2[i, j] = int(m[0]*0.3 + m[1]*0.59 + m[2]*0.11)
plt.imshow(img2, cmap='gray')

# 二值图
plt.subplot(223)
img3 = rgb2gray(img)
for i in range(img3.shape[0]):
    for j in range(img3.shape[1]):
        if img3[i, j] <= 0.5:
            img3[i, j] = 0
        else:
            img3[i, j] = 1
plt.imshow(img3, cmap='gray')
plt.show()


