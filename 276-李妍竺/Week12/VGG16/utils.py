import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops #类似np.array

def load_image(path):
    # 读取图片，并将图片修剪成正方形
    img = mpimg.imread(path) #rgb
    short_edge = min(img.shape[:2])
    h = int((img.shape[0] - short_edge) / 2)
    w = int((img.shape[1] - short_edge) / 2)
    crop_img = img[h:h+short_edge,w:w+short_edge]
    return crop_img
'''
def resize_image(image,size):  #这里的size是 h w
    with tf.name_scope('resize_image'): #tf.name_scope 给变量包一层名字 'resize_image' 是提供的范围名称。它用于为变量或操作命名，以帮助识别和理解代码。
        images = []
        for i in image:
            i = cv2.resize(i,size)
            images.append(i)
        images = np.array(images)
        return images
'''
def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size,
                                       method, align_corners)
        image = tf.reshape(image, tf.stack([-1,size[0], size[1], 3]))  #沿轴堆叠，-1所在轴数量可变，沿其堆叠
        return image


def print_prob(prob,file_path):
    synset = [l.strip() for l in open(file_path).readlines()] #删除每行开头，末尾的空白字符

    # 将概率从大到小排列的结果的序号存入pred
    pred = np.argsort(prob)[::-1]  # [::-1] 倒序  list(reversed())   返回的是下标

    # 取最大的1个、5个。
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))

    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))


    return top1


#print_answer(0)



