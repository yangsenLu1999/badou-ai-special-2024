import numpy as np
import tensorflow as tf
import cv2
import matplotlib.image as mpimg

def load_image(path):
    img = mpimg.imread(path)
    #把图片剪切成中心正方形
    short_eage = min(img.shape[:2])
    h = int((img.shape[0]-short_eage)/2)
    w = int((img.shape[1]-short_eage)/2)
    img_crop = img[h:h+short_eage,w:w+short_eage]
    return img_crop


def resize_image(image,size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i,size)
            images.append(i)
        images = np.array(images)
        return images

def print_answer(argmax):
    with open('./data/model/index_word.txt','r',encoding='utf-8') as f:
        l = f.readlines()
        answer = l[argmax].split(';')[-1]
        print(answer)
        return answer

