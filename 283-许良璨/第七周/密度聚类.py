import sklearn.datasets
import sklearn.cluster.DBSCAN
import numpy as np
import matplotlib.pyplot as plt

d=datasets.lord_iris()
x=d.data[:,:4]

l=DBSCAN(eps=0.5,min_samples=9)
l.fix(x)
label=l.labels_

x0=x[label==0]
x1=x[label==1]
x2=x[label==2]

plt.scatter(x0[:,0],x0[:,1],c="red",marker="o",label="label0")
plt.scatter(x1[:,0],x1[:,1],c="blue",marker="*",label="label1")
plt.scatter(x2[:,0],x2[:,1],c="green",marker="#",label="label2")

plt.legend(loc="0")
plt.show()
