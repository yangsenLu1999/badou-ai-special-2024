import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import array_ops

tf.disable_v2_behavior()
# 将图片变成一个正方形，长的一个边超出的部分修剪掉
def load_image(path):
    img=mpimg.imread(path)
    #将图片变成一个正方形，长的一个边超出的部分修剪掉
    short_edge = min(img.shape[:2])   #找到短边
    y=(img.shape[0]-short_edge)//2
    x=(img.shape[1]-short_edge)//2
    result=img[y:y+short_edge,x:x+short_edge]
    return result

#缩放图像
def resize_image(image, size):
    with tf.name_scope('resize_image'):#表示创建了一个名为 "resize_image" 的命名空间，所有在这个命名空间内创建的操作都会被自动命名为 "resize_image/操作名" 的形式
        images = []
        for i in image:
            i = cv2.resize(i, size)  #用于缩放图像，cv2.resize 函数还可以接受一个插值方法的参数，用于指定图像缩放时的插值方法，
            images.append(i)
        images = np.array(images)
        return images

#输出标签
def print_answer(argmax):
    with open("./index_word.txt","r",encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]  #l.split(";")[1][:-1] 最后的[:-1旨在去除\n这个元素]
    print(synset[argmax])
    return synset[argmax]