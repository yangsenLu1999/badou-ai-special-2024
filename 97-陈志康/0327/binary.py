import cv2
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# 读取图片
image = cv2.imread('lenna.png')
# print('===cv2====')
# print(image)
# print('===plt====')
# print(plt.imread('lenna.png'))
# exit()
# 先转成正常RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 获取图片宽高
h, w = image.shape[:2]
print(h, w)
# # 初始化一张相同规格的空图片
image_gray = np.zeros([h, w], image.dtype)
# 图片灰度化
# cv2封装的灰度化函数
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # 转成正常RGB
# image_gray = rgb2gray(image)
image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2RGB)
# print('image_gray 值:%s' % image_gray)


plt.subplot(221)  # 画布布局221位置
plt.imshow(image)  # 直接显示原图
# cv2.imshow('show image_gray', image_gray)
# cv2.waitKey(0)

plt.subplot(222)  # 画布布局222位置
plt.imshow(image_gray, cmap='gray')  # 显示灰度图

print('---image gray----')
print(image_gray)

img_gray = rgb2gray(image)
plt.subplot(224)  # 画布布局224位置
plt.imshow(img_gray, cmap='gray')  # 显示老师灰度图

print('---laoshi image gray----')
print(img_gray)

# 打印结果image_gray数据结构是3个层级，而老师的img_gray数据结构是2个层级

# 二值化
# 初始化一张相同规格的空图片
image_binary = np.zeros([h, w], img_gray.dtype)
rows, cols = img_gray.shape
# 遍历每个坐标点
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] <= 0.5:
            image_binary[i, j] = 0
        else:
            image_binary[i, j] = 1
print('result')
print('binary image 值:%s' % image_binary)
plt.subplot(223)  # 画布布局223位置
plt.imshow(image_binary, cmap='gray')
plt.show()
