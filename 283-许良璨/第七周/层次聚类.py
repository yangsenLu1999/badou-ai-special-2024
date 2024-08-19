import matplotlib.pyplot as plt
from sicpy.cluster.hierarchy import dendrogram,fcluster,linkage




x=[[5,1],
   [9,8],
   [5,5],
   [7,10],
   [4,3]]

l=linkage(x,"ward")
f=fcluster(l,4,"distance")
fig=plt.figure(figsize(5,3))
d=dendrogram(l)
plt.show()
