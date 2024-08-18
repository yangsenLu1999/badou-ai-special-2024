import cv2
import numpy as np
import skimage
import matplotlib.pyplot as plt

# 图片灰度化
# 读取图片
image = cv2.imread('lenna.png')
# 获取图片宽高
h, w = image.shape[:2]
print(h, w)
# 初始化一张相同规格的空图片
image_gray = np.zeros([h, w], image.dtype)
# 手写灰度化功能
# 遍历原图的RGB值并灰度化
# for i in range(h):
#     for j in range(w):
#         m = image[i, j]  # 获取平面点坐标
#         image_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 点坐标对应的RGB值乘以各系数，灰度化
# cv2封装的灰度化函数
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('result')
print('image_gray 值:%s' % image_gray)
cv2.imshow('show image_gray', image_gray)
cv2.waitKey(0)
