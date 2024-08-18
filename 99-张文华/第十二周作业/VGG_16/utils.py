import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops


def load_image(path):
    image = mpimg.imread(path)
    # 将图像修剪成中心的正方形
    short_edge = min(image.shape[:2])
    xx = int((image.shape[0]-short_edge) / 2)
    yy = int((image.shape[1]-short_edge) / 2)
    crop_img = image[yy:yy+short_edge, xx:xx+short_edge]
    return crop_img

#
def resize_img(image, size, method=tf.image.ResizeMethod.BILINEAR,
               align_corners=False):
    image = tf.expand_dims(image, dim=0)
    image = tf.image.resize_images(image, size, method, align_corners)
    image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
    return image


#
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    pred = np.argsort(prob)[::-1]

    # 取最大的一个
    top1 = synset[pred[0]]
    print('top1', top1, prob[pred[0]])

