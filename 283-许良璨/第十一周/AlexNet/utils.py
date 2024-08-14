import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import cv2

def load_image(path):
    img=mpimg.imread(path)
    shortedge=min(img.shape()[0],img.shape()[1])
    yy=int((img.shape()[0]-shortedge)/2)
    xx=int((img.shape()[1]-shortedge)/2)
    crop_img=img[yy:yy+shortedge,xx:xx+shortedge]
    return crop_img

def resize_img(img,size):
    with tf.name_scope("resize image"):
        images=[]
        for i in img:
            i=cv2.resize(i,size)
            images=images.append(i)
        images=np.array(images)
    return images

def print_answer(argmax):
    with open("./data/model/index_word.txt","r",encoding="utf-8") as f:
        answer=[l.split(";")[1][:-1] for l in f.readlines()]
    print(synset[argmax])
    return answer[argmax]