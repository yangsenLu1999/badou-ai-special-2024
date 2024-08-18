import numpy as np;
import cv2;

# 近邻插值法
def function(img):
    #获取长宽和通道
    height,weight,channel = img.shape;
    print(channel)
    #创建800 * 800 为0的图像数组
    emptImg = np.zeros((800, 800,channel), img.dtype)
    sh = 800/height;
    sw = 800/weight;
    for i in range(800):
        for j in range(800):
            x = int(i/sh + 0.5);
            y = int(j/sw + 0.5);
            emptImg[i,j] = img[x,y];
    return emptImg;

img = cv2.imread("lenna.png")
zoom = function(img);
# print(zoom)
# print(zoom.shape)
cv2.imshow("nearest interp", zoom);
cv2.imshow("img", img);
cv2.waitKey(0);
cv2.destroyAllWindows();


# 将图像调整到 800x800 像素
# 第二个参数是目标图像大小，以 (宽度, 高度) 的形式给出
# 第三个参数是插值方法，这里可以使用 cv2.INTER_NEAREST 或 cv2.INTER_LINEAR
resized_img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)

# 显示调整大小后的图像
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


