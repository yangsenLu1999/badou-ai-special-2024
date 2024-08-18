
# 图像灰度化，二值化
import cv2
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


img = cv2.imread("lenna.png")
height,weight = img.shape[:2];
print("长{},高{}".format(height,weight))
# print("长%s,高%s" %(height,weight))
#创建一张空白图片
img_grayx = np.zeros((height,weight), img.dtype);

#灰度
for i in range(height):
    for j in range(weight):
        m = img[i,j]
        img_grayx[i,j] = int(m[0] * 0.11 + m[1]*0.59 + m[2] * 0.3)

# cv2.imshow("img gray", img_grayx)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.subplot(2,2,1);
img = plt.imread("lenna.png")
plt.imshow(img)

#二值化
plt.subplot(2,2,2);
img = plt.imread("lenna.png")
img_gray = rgb2gray(img)
for i in range(height):
    for j in range(weight):
        if (img_gray[i,j] <=0.5):
            img_gray[i,j] = 0
        else:
            img_gray[i, j] = 1

plt.imshow(img_gray, cmap="gray")
plt.axis("off")

#二值化
plt.subplot(2,2,3);
img = plt.imread("lenna.png")
img_gray = rgb2gray(img)
img_binary= np.where(img_gray >=0.5, 1, 0)
plt.imshow(img_binary, cmap="gray")

#灰度
plt.subplot(2,2,4);
img = cv2.imread("lenna.png")
height,weight = img.shape[:2];

#创建一张空白图片
img_gray = np.zeros((height,weight), np.uint8);

#灰度
for i in range(height):
    for j in range(weight):
        m = img[i,j]
        img_gray[i,j] = int(m[0] * 0.3 + m[1]*0.59 + m[2] * 0.11)

plt.imshow(img_gray, cmap="gray")

plt.show()