import numpy as np
import cv2


def function(img, width2, high2):
    width, high, channels = img.shape
    emptyimg = np.zeros((width2, high2, channels), np.uint8)
    sh = high2 / high
    sw = width2/width
    for i in range(width2):
        for j in range(high2):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            emptyimg[i, j] = img[x, y]
    return emptyimg


img = cv2.imread("c4b591762800e7b417922ee4bcfb4cd.jpg")
# cv2.imshow("原图导入", img)
# cv2.waitKey(0)
print(img.shape)
width2 = int(input("请输入放大后的长"))
high2 = int(input("请输入放大后的宽"))
newimg = function(img, width2, high2)
cv2.imshow("newimg", newimg)
cv2.waitKey(0)