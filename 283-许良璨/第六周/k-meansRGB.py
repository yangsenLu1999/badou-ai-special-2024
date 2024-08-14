import cv2
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread(../lenna.png)

data=img.reshape((-1,3))
data=np.float32(data)

criteria=[cv2.TERM_CRITERIA_EPS]+[cv2.TERM_CRITERIA_MAX_ITER,10,1]

flag=cv2.KMEANS_RANDOM_CENTERS

compectness,labels,centers=cv2.kmeans(data,8,None,criteria,10,flag)

labels=np.uint8(labels)
res=centers[labels.flatten()]
res=res.reshape(img.shape)

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
res=cv2.cvtColor(res,cv2.COLOR_BGR2RGB)

IMG=[img,res]
title=["原图","压缩图"]
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(IMG[i],"gray")
    plt.title[title[i]]

plt.show()
