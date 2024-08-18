import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("lenna.png")
h,w = img.shape[:2]

def floating_point_gray():
    img_gray = np.zeros([h,w],img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            #灰度图像素赋值 gray=(R*0.3+G*59+B*0.11)
            img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
    return img_gray



img_floating = floating_point_gray()

#图像最大值灰度处理
def max_gray():
    img_gray = np.zeros([h,w],img.dtype)
    for i in range(h):
        for j in range(w):
            #获取图像R G B最大值
            gray = max(img[i,j][0], img[i,j][1], img[i,j][2])
            #灰度图像素赋值 gray=max(R,G,B)
            img_gray[i,j] = np.uint8(gray)
    return img_gray


img_max = max_gray()
#图像整数灰度处理
def integer_gray():
    img_gray = np.zeros([h,w],img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            #灰度图像素赋值 gray=(R*30+G*59+B*11)/100
            img_gray[i,j] = int(m[0]*11 + m[1]*59 + m[2]*30)/100
    return img_gray

img_integer = integer_gray()



#图像整数灰度处理
def displacement_gray():
    img_gray = np.zeros([h,w],img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            #灰度图像素赋值 gray=(R*28+G*151+B*77)>>8
            img_gray[i,j] = int(m[0]*28 + m[1]*151 + m[2]*77)>>8
    return img_gray

img_displacement = displacement_gray()

def average_gray():
    img_gray = np.zeros([h,w],img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            #灰度图像素赋值 gray=(R+G+B)/3
            img_gray[i,j] = int(m[0] + m[1] + m[2])/3
    return img_gray

img_average = average_gray()

def weighted_gray():
    img_gray = np.zeros([h,w],img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            #灰度图像素赋值 gray=(R+G+B)/3
            img_gray[i,j] = int(m[0]*0.144 + m[1]*0.587 + m[2]*0.299)
    return img_gray

img_weighted = weighted_gray()

#opencv 的实现
r, b = cv2.threshold(img_floating, 127, 255, cv2.THRESH_BINARY)

def binarize_image(image):
    gray_image = floating_point_gray() #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_value = np.mean(gray_image)
    binary_image = np.where(gray_image > mean_value, 255, 0).astype(np.uint8)
    return binary_image

img_binary = binarize_image(img)


def zoom(img, scale):
    height, width, channels = img.shape

    new_height = int(height * scale)
    new_width = int(width * scale)
    new_img = np.zeros([new_height, new_width, channels], np.uint8)
    for h in range(new_height):
        for w in range(new_width):
            x = int(h / scale)
            y = int(w / scale)
            new_img[h, w] = img[x, y]
    return new_img


def nearest_neighbor(image, factor):
    h, w = image.shape[:2]
    
    new_h = int(h * factor)
    new_w = int(w * factor)
    
    new_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    
    # 计算插值后每个像素在原始图像中的位置，并进行最临近插值
    for i in range(new_h):
        for j in range(new_w):
            x = int(i / factor)
            y = int(j / factor)
            new_image[i, j] = image[x, y]
    
    return new_image

resized_image = nearest_neighbor(img, 1.5)

titles = ['floating gray', 'max gray', 'integer gray', 'displacement gray', 'average gray',
          'weighted gray', 'binary image(OPENCV)', 'binary image(me)', 'resized image']
images = [img_floating, img_max, img_integer, img_displacement, img_average,
          img_weighted, b, img_binary, resized_image]

for i in range(9):
    plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()