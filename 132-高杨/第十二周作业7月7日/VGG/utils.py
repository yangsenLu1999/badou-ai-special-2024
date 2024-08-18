import matplotlib.image as mpimg
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def load_image(path):
    # 读取图片 rgb
    img = mpimg.imread(path)

    # 将图片修建成中心正方形
    short_img = min(img.shape[:2])

    yy = int((img.shape[0] - short_img) / 2)
    xx = int((img.shape[1]- short_img) / 2)
    clip_img = img[yy:yy+short_img,xx:xx+short_img]

    return clip_img

def resize_img(img,size,method = tf.image.ResizeMethod.BILINEAR,
               align_corners=False
               ):
    with tf.name_scope('resize_img'):
        img = tf.expand_dims(img,0)
        img = tf.image.resize_images(img,size,method,align_corners)
        img = tf.reshape(img,tf.stack([-1,size[0],size[1],3]))

        print(img.shape)


        return img


def print_prob(prob,file_path):
    synset = [l.strip()  for l in open(file_path).readlines()]
    pred = np.argsort(prob)[::-1]

    # 取可能最大的前1个，前5个
    top1 = synset[pred[0]]
    print(('top 1：',top1,prob[pred[0]]))

    top5 = [(synset[pred[i]] , prob[pred[i]])  for i  in range(5)]

    print(("top5:",top5))

    return top1









