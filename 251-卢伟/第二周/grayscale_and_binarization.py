# 读取图片
import numpy as np
import cv2


def process_pic():
    # 读取图像
    img = cv2.imread("C:\\Users\\1\\Desktop\\test1.png")
    # 获取图像heigh 和 weight
    h, w = img.shape[:2]
    # new 一个新图像，大小和当前图像一样
    grey_result = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            # 获取3个通道的数组
            array = img[i, j]
            # cv是GBR 灰度公式 G*0.11 + B*0.59 + R*0.3
            grey_result[i, j] = array[0] * 0.11 + array[1] * 0.59 + array[2] * 0.3
    # 保存灰度化结果
    # cv2.imwrite("C:\\Users\\1\\Desktop\\gray.png", grey_result)
    cv2.imshow("grayscale", grey_result)
    cv2.waitKey()

    # 二值化
    h, w = grey_result.shape
    result = np.zeros((h, w), dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            if grey_result[i, j] > 127:
                result[i, j] = 255  # 255是白色
            else:
                result[i, j] = 0  # 0是黑色
    # 保存二值化结果
    cv2.imshow("binarization", result)
    cv2.waitKey()
    # cv2.imwrite("C:\\Users\\1\\Desktop\\binarization.png", result)


# 插值
process_pic()
