import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# method a
image_d = plt.imread("picture/lenna.png")

h, w = image_d.shape[:2]

gray_image_d = np.zeros([h, w], image_d.dtype)

for i in range(h):
    for j in range(w):
        m = image_d[i, j]
        gray_image_d[i, j] = m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3

image_binary = np.where(gray_image_d >= 0.5, 1, 0)

print("-----imge_binary------")
print(image_binary)
print(image_binary.shape)

plt.imshow(image_binary, cmap='gray')
plt.show()


# method b
image_b = plt.imread("picture/lenna.png")

gray_image_b = rgb2gray(image_b)

image_binary = np.where(gray_image_b >= 0.5, 1, 0)

plt.imshow(image_binary, cmap='gray')

plt.show()