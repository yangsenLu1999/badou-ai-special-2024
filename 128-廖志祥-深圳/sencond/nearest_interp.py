import cv2
import numpy as np


def nearest(image, nh, nw):
    h, w, c = image.shape
    empty_image = np.zeros((nh, nw, c), np.uint8)
    sh = nh / h
    sw = nw / w

    for i in range(nh):
        for j in range(nw):
            x = round(i / sh)
            y = round(j / sw)

            empty_image[i, j] = image[x, y]

    return empty_image


height = 960
weight = 960

image = cv2.imread("picture/lenna.png")
new_pic = nearest(image, height, weight)

cv2.imshow("nearest interp", new_pic)
cv2.imshow("image", image)
cv2.waitKey(0)

#  问： 为什么长宽设置大于1024会报错？
