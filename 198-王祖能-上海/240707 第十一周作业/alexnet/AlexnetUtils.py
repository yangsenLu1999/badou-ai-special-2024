# 准备处理图像（裁剪、缩放尺寸），按序号打印标签等功能
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg


def load_img(path):
    img = mpimg.imread(path)  # RGB格式读取图片
    h, w, c = img.shape[:]
    short_edge = min(h, w)  # min(img.shape[:2]
    x_delta = int((w - short_edge) / 2)
    y_delta = int((h - short_edge) / 2)
    crop_img = img[y_delta: y_delta + short_edge, x_delta: short_edge + x_delta]  # 将图片按短边尺寸修剪成中心的正方形
    return crop_img


def resize_img(img, size):
    with tf.name_scope('resize_img'):  # 因为img不是一张图，可能传入一系列batch的图像，需要逐个处理后添加到一个空集合汇总
        img_new = []
        for i in img:
            i = cv2.resize(i, size)
            img_new.append(i)
        img_new = np.array(img_new)  # 将空集中不断追加改变尺寸后的图片，并强制转array格式
        return img_new


def print_answer(argmax):
    with open('../../alexnet/AlexNet-Keras-master/data/model/index_word.txt', 'r', encoding='utf-8') as f:
        # synset = [I.split(';')[1] for I in f.readlines()]  # 结果是['猫\n', '狗\n']
        synset = [l.split(';')[1][:-1] for l in f.readlines()]  # 过滤掉换行符，每个元素取出除最后换行符的字符串
    print(synset)
    return synset[argmax]  # synset是一个[猫， 狗]的列表，根据推理的结果概率排序，选出最大概率0， 1索引序号对应的名称。
