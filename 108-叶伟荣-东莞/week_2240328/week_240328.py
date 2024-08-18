import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.color import rgb2gray

# 临时更改字体,避免中文显示乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

img = cv.imread('data/demo.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)


def gray_show():
    """
    灰度化以及二值化显示
    Gray = R*0.3+G*0.59+B*0.11
    :return: 灰度化图片
    """
    gray_x = np.array([0.3, 0.59, 0.11])
    # 方法一 向量点积
    gray_img1 = img_rgb[:, :].dot(gray_x)
    # 方法二 直接调用api
    gray_img2 = rgb2gray(img)

    # 二值化
    gray_binary = np.where(gray_img1 / 255 < 0.5, 0, 1)

    # 使用matplotlib显示，需要转成rgb显示
    plt.subplot(221)
    plt.imshow(img_rgb)
    plt.title('原图')
    plt.subplot(222)
    plt.imshow(gray_img1, cmap='gray')
    plt.title('向量点积灰度化')
    plt.subplot(223)
    plt.imshow(gray_img2, cmap='gray')
    plt.title('skimage方法调用灰度化')
    plt.subplot(224)
    plt.imshow(gray_binary, cmap='gray')
    plt.title('二值化')
    plt.tight_layout()
    plt.show()


# 灰度化以及二极化显示
gray_show()


def zoom_image(height, width, original_img):
    """
    图片缩放，最临近插值
    :param height: 设置想要放大的高度
    :param width: 设置想要放大的度
    :param original_img: 原图 (需要rgb格式，非bgr)
    """
    zoom_img = np.zeros([height, width, original_img.shape[2]], dtype=original_img.dtype)
    height_rate = original_img.shape[0] / height
    width_rate = original_img.shape[1] / width
    for i in range(height):
        # 减 1 以免造成 超出索引范围
        actual_i = int(i * height_rate + 0.5) - 1
        for j in range(width):
            actual_j = int(j * width_rate + 0.5) - 1
            zoom_img[i, j] = original_img[actual_i, actual_j]
    plt.subplot(121)
    plt.imshow(original_img)
    plt.title('原图')
    plt.subplot(122)
    plt.imshow(zoom_img)
    plt.title('缩放的高%dx宽%d' % (height,width))
    plt.tight_layout()
    plt.show()


# 图片缩放，最临近插值
# 放大 （不要太大，分辨率越大，就越慢慢）
zoom_image(1200, 1200, img_rgb)

# 缩小
# zoom_image(400,400, img_rgb)
