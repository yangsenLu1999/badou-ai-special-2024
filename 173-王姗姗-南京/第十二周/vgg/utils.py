import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops


def load_image(path):
    # 加载图片
    img = mpimg.imread(path)
    # 将图片修改为正方形
    short_img = min(img.shape[:2])  # 获取图片行数和列数最小值
    row = int((img.shape[0] - short_img) / 2)  # 计算图片在行方向上的偏移量
    column = int((img.shape[1] - short_img) / 2)  # 计算图片在列方向上的偏移量
    children_img = img[row:row + short_img, column:column + short_img]  # 在原图像上截取指定大小的子图像
    return children_img


def resize_image(img, size,
                 method=tf.image.ResizeMethod.BILINEAR,  # 使用双线性插值方法对图像进行缩放
                 align_corners=False):
    with tf.name_scope('resize_image'):
        # 在索引为0的位置添加一个维度
        image = tf.expand_dims(img, 0)
        # 调用tf.image.resize_images函数对图像进行缩放
        image = tf.image.resize_images(image, size, method, align_corners)
        # 删除索引为0的位置的维度
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
        return image


def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    # 预测结果
    pred = np.argsort(prob)[::-1]
    # 取最大的1个、5个。
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
