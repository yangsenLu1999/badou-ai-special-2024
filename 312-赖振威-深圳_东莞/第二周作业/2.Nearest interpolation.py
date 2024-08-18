import cv2
import numpy as np

# Nearest interpolation
# 定义图像放缩函数
def function(img):
    # 获取原始图像的高度、宽度和通道数信息
    height, width, channels = img.shape   # 获取原图像的数组的维度信息，包括高度、宽度和通道数(512, 512, 3)
    # print('----img.shape----\n',img.shape)
    # 创建一个空白图像，尺寸为 800x800，与原始图像具有相同的通道数
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    # print(emptyImage)

    # # 显示图像
    # cv2.imshow("Empty Image", emptyImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 计算高度和宽度的缩放比例
    sh = 800 / height
    sw = 800 / width

    # 遍历新图像的每个像素  找到坐标最近的像素点赋值
    for i in range(800):
        for j in range(800):
            # 根据缩放比例将新图像中的像素坐标四舍五入的方法来进行坐标映射回原图像中的坐标  从而更加准确地映射新旧像素坐标
            x = int(i / sh + 0.5)  # 将 i 坐标映射回原图像中的 x 坐标   取最近坐标的像素点
            y = int(j / sw + 0.5)  # 将 j 坐标映射回原图像中的 y 坐标

            # 将原图像中对应位置的像素值复制到新图像中
            emptyImage[i, j] = img[x, y]

    # 返回放缩后的图像
    return emptyImage

# 读取图像
img = cv2.imread("lenna.png")

# 使用自定义函数进行放缩
zoom = function(img)

# # 输出新图像的信息
# print('---zoom----')
# print(zoom)  # 不同坐标上取得不同的像素点
# print('---zoom.shape----')
# print(zoom.shape)   # (800, 800, 3)

# 显示放缩后的图像和原图像
cv2.imshow("nearest interp", zoom)
cv2.imshow("original image", img)
cv2.waitKey(0)
