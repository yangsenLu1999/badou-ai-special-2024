import numpy as np

def normalization1£¨x£©:
    return[(float(i)-min(x))/(max(x)-min(x)) for i in x]

def normalization2£¨x£©:
    return[(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]

def normalization3£¨x£©:
    s1=sum((float(i)-np.mean(x))*(float(i)-np.mean(x)) for i in x)/len(x)
    return[(float(i)-np.mean(x))/s1 for i in x]