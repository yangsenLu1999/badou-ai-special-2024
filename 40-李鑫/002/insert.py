import cv2
import numpy as np

# 读取图片
img = cv2.imread('lenna.png')
# 获取图片高、宽
h, w, c = img.shape
# 创建一张新的放大图片
img_bigger = np.zeros([800, 800, c], np.uint8)
for i in range(800):
    for j in range(800):
        img_bigger[i, j] = img[int(h/800*i), int(w/800*j)]

cv2.imshow("bigger", img_bigger)
cv2.waitKey(0)

