import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

def load_image(path):
    #根据图片的尺寸对图片以按中心修剪的方式修剪成正方形
    img=mpimg.imread(path)
    b=min(img.shape[0:2])
    x=int((img.shape[1] - b )/2)
    y=int((img.shape[0] - b )/2)
    img_copy=img[x:x+b,y:y+b]
    return img_copy

def resize_image(image,
                 size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False
    ):
    with tf.name_scope('resize_image'):
        image=tf.expand_dims(image,0)
        image=tf.imag.resize_image(image,size,method,align_corners)
        image=tf.reshape(image,tf.stack([-1,size[0],size[1],3]))
    return image

def print_prob(prob,file_path):

    synset=[l.strip() for l in open(file_path).readlines()]

    pred=tf.argsort(prob)[::-1]
    print(pred)

    top1=synset[pred[0]]
    print("top1:",top1,prob[pred[0]])
    top5=[(synset[pred[i]],prob[pred[i]])for i in range(5)]
    print("top5",top5)
    return top1