from PIL import Image
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2
import numpy as np

# method a

image_a = Image.open("picture/lenna.png")

gray_image_a = image_a.convert("L")

gray_image_a.save("picture/gray_lenna_a.png")

gray_image_a.show()

# method b

image_b = plt.imread("picture/lenna.png")

gray_image_b = rgb2gray(image_b)

plt.imshow(gray_image_b, cmap="gray")

plt.show()

# method c

image_c = cv2.imread("picture/lenna.png")

gray_image_c = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)

plt.imshow(gray_image_c, cmap="gray")

plt.show()

# method d

image_d = cv2.imread("picture/lenna.png")

h, w = image_d.shape[:2]

gray_image_d = np.zeros([h, w], image_d.dtype)

for i in range(h):
    for j in range(w):
        m = image_d[i, j]
        gray_image_d[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)

plt.imshow(gray_image_d, cmap="gray")
plt.show()

