import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
def read_image(path):
    # 读取图片rgb
    image_one = mpimg.imread(path)
    # 将图片修剪成正方形
    min1 = min(image_one.shape[:2])
    y_y = int((image_one.shape[0] - min1) / 2)
    x_x = int((image_one.shape[1] - min1) / 2)
    copy_image = image_one[y_y: y_y + min1, x_x: x_x + min1]
    return copy_image


def resize_image(image, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size, method, align_corners)
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
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
