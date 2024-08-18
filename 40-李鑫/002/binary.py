import cv2

# 读取图片
img = cv2.imread('lenna.png')
# 创建一张新的等比例图片
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 获取图片高、宽
h, w = img_gray.shape
# 遍历原图片像素点
for i in range(h):
    for j in range(w):
        if img_gray[i, j] / 255 <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 255

cv2.imshow('binary', img_gray)
cv2.waitKey(0)