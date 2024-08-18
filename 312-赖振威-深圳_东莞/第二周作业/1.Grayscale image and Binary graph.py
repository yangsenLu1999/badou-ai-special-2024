# -*- coding: utf-8 -*-

import time
# 导入所需的模块
from skimage import io, color      # 导入 skimage 库中的 io 模块和 color 模块
from skimage.color import rgb2gray  # 从 skimage 库中导入 rgb2gray 函数，用于将彩色图像转换为灰度图像
import numpy as np  # 导入 NumPy 库，用于处理数组和矩阵运算
import matplotlib.pyplot as plt  # 导入 matplotlib 库中的 pyplot 模块，并将其重命名为 plt，用于绘图
from PIL import Image  # 导入 PIL 库中的 Image 模块，用于图像处理
import cv2  # 导入 OpenCV 库，用于图像处理和计算机视觉任务

# Grayscale image 方法一
def func1():
    # 灰度化
    img = cv2.imread("lenna.png")   # 读取图片 imread该函数会返回一个表示图像BRG 通道的多维数组  返回一个整数数组，表示图像的像素值。
    # print(img)
    h,w = img.shape[:2]             # 获取图片的high和wide
    # print(img.shape)                # (512, 512, 3)

    # img_gray = np.zeros([h,w,3],img.dtype)
    img_gray = np.zeros([h,w],img.dtype)   #创建一张和当前图片大小一样的黑色单通道图片 矩阵512*512*0
    cv2.imshow("image show gray", img_gray)  # 展示原始图
    cv2.waitKey(0)  # 等待用户按下任意键
    cv2.destroyAllWindows()  # 关闭所有窗口

    # 灰度化img_gray图像
    for i in range(h):  # 遍历图像的每一行
        for j in range(w):  # 遍历图像的每一列
            m = img[i, j]  # 获取坐标位置 (i, j) 处的像素值，即 BGR 值  取出当前high和wide中的BGR坐标

            # 将 BGR 值转换为灰度值，并赋值给新图像的对应位置
            img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)   # BRG乘上的数值都是科学家调查适合人眼的灰度图数值 没有公式

    # # 打印灰度化后的图像数组
    # print("image show gray: \n%s" % img_gray)

    # 显示灰度化后的图像
    cv2.imshow("image show gray", img_gray)
    cv2.waitKey(0)  # 等待用户按下任意键

# func1()

# Grayscale image 方法二
def func2():
    # plt处理
    plt.subplot(121)  # 设置子图布局为 2x2，当前子图位置为第一格
    '''
    Matplotlib 读取的图像数据会进行数值归一化处理，即将像素值从整数范围（比如 0 到 255）映射到 0 到 1 的浮点数范围。
    这种归一化处理有助于统一不同图像的像素表示方式，并且在计算机中更容易进行处理和显示
    '''
    img = plt.imread("lenna.png")  # plt 模块来读取名为 "lenna.png" 的图像文件  返回一个浮点数数组，表示图像的像素值
    # 设置子图布局为 1x2，当前子图位置为第一格
    plt.imshow(img)  # 将读取的图像数据 img 显示在当前的 Matplotlib 图形中
    plt.title('Original Image')

    # 灰度化
    # img_gray = rgb2gray(img)   # 接口一
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 接口二
    plt.subplot(122)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Grayscale Image')

    plt.tight_layout()
    plt.show()

func2()

# Binary graph 方法一
def func3():
    # plt处理
    plt.subplot(221)  # 设置子图布局为 2x2，当前子图位置为第一格
    '''
    Matplotlib 读取的图像数据会进行数值归一化处理，即将像素值从整数范围（比如 0 到 255）映射到 0 到 1 的浮点数范围。
    这种归一化处理有助于统一不同图像的像素表示方式，并且在计算机中更容易进行处理和显示
    '''
    img = plt.imread("lenna.png")  # plt 模块来读取名为 "lenna.png" 的图像文件  返回一个浮点数数组，表示图像的像素值
    # 设置子图布局为 1x2，当前子图位置为第一格
    plt.imshow(img)  # 将读取的图像数据 img 显示在当前的 Matplotlib 图形中
    plt.title('Original Image')

    # 灰度化
    img_gray = rgb2gray(img)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = img
    plt.subplot(222)
    # plt.show()
    plt.imshow(img_gray, cmap='gray')
    plt.title('Grayscale Image')
    print("---image gray----")
    print(img_gray)

    # 二值化
    print(img_gray.shape)   # (512, 512)
    rows, cols = img_gray.shape
    for i in range(rows):
        for j in range(cols):
            if (img_gray[i, j] <= 0.5):
                img_gray[i, j] = 0
            else:
                img_gray[i, j] = 1

    img_binary = np.where(img_gray >= 0.5, 1, 0)
    print("-----imge_binary------")
    print(img_binary)
    print(img_binary.shape)  # (512, 512)

    plt.subplot(223)
    plt.imshow(img_binary, cmap='gray')
    plt.title('Binary Image')
    # # 调整第三张图的位置，使其向下移动一些，可以通过增加 bottom 参数的值来实现
    # plt.subplots_adjust(top=0.85, bottom=0.55)
    # 使用 tight_layout() 自动调整子图布局
    plt.tight_layout()
    plt.show()

func3()
