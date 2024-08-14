import cv2
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread("../lenna.png")
k=64
cv2.imshow("",img)
cv2.waitKey(0)


data=img.reshape(-1,img.shape[2])
data=np.float32(data)

criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flag=cv2.KMEANS_RANDOM_CENTERS

conpectness,labels,centers=cv2.kmeans(data,k,None,criteria,10,flag)

centers=np.uint8(centers)
res=centers[labels.flatten()]
dst=res.reshape((img.shape))


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
imgs=[img,dst]
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(imgs[i],"gray")
    plt.xticks([]), plt.yticks([])

plt.show()