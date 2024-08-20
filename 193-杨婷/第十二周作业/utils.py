import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops


def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img


# 调整图片形状
def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,  # method：调整图像大小的方法，默认为双线性插值
                 align_corners=False):  # 指示是否在调整大小时将图像的角落对齐,当为 False 时，缩放是相对于图像中心进行的
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)  # 在image的最前面增加了一个维度，使其形状变为[1, height, width, channels]
        image = tf.image.resize_images(image, size,
                                       method, align_corners)
        # image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
        return image


def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    # 将概率从大到小排列的结果的序号存入pred
    pred = np.argsort(prob)[::-1]
    # 取最大的1个、5个。
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1



