import cv2
import numpy as np

# 读取图片
img = cv2.imread('lenna.png')
# 获取图片高、宽
h, w = img.shape[:2]
# 创建一张新的等比例图片
img_gray = np.zeros([h, w], img.dtype)
# 遍历原图片像素点
for i in range(h):
    for j in range(w):
        # 获取原图像素点的bgr值
        b, g, r = img[i, j][:3]
        img_gray[i, j] = int(r * 0.3 + g * 0.59 + b * 0.11)
cv2.imshow('gray', img_gray)
cv2.waitKey(0)
