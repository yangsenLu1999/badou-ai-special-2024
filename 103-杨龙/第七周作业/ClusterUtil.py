# -*- coding:utf-8 -*-

__author__ = 'Young'

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

# 相关方法介绍
'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法，常见method值
single(最近邻)|complete(最远邻)|average(平均距离)|weighted(加权平均距离)|centroid(质心距离)|median(中位数距离)|attan(曼哈顿距离)...
3. linkage方法进行层次聚类，生成聚类结果的链接矩阵，结果包含：
    count：当前聚类的样本数量。
    dist：当前聚类的距离。
    ind1：第一个聚类的索引。
    ind2：第二个聚类的索引。
    order：表示当前聚类是由哪个聚类分裂而来的，如果是第一次聚类，则为0
'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1. 第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2. t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
3. criterion用于生成扁平聚类的标准：
    inconsistent：如果一个聚类节点及其所有后代的不一致性值小于等于t，则它们的所有叶子后代属于同一个扁平聚类。如果没有符合此条件的非单例聚类，则每个节点被分配到其自己的聚类（默认）。
    distance：形成扁平聚类，使每个扁平聚类中的原始观测之间的共生距离不大于t。
    maxclust：查找最小阈值r，使得同一扁平聚类中任意两个原始观测之间的共生距离不大于r，且形成的扁平聚类不超过t。
    monocrit：当monocrit[j]小于等于t时，从索引为i的聚类节点c形成扁平聚类。例如，要在不一致矩阵R中计算的最大平均距离上设置阈值0.8，可以使用：MR = maxRstat(Z, R, 3)然后调用fcluster(Z, t=0.8, criterion='monocrit', monocrit=MR)。
    maxclust_monocrit：从非单一群集节点形成扁平聚类c，当对于所有聚类索引i以下（包括）c，有monocrit[i] <= r。r被最小化，使得形成的扁平聚类不超过t。monocrit必须是单调的。
4. depth：执行不一致性计算的最大深度，对其他标准没有意义，默认值为2。
5. R：用于'inconsistent'标准的不一致矩阵，如果未提供，则计算该矩阵。
6. monocrit：长度为n-1的数组，用于对非单例进行阈值处理的统计量。monocrit向量必须是单调的。
'''


def hierarchical_clustering(matrix, method, t, criterion):
    """
    可以尝试各种组合，避免写重复的代码
    :param matrix: 原始数据
    :param method: 层次聚类距离计算方式
    :param t: 聚类阈值
    :param criterion: 生成扁平聚类标准
    :return:
    """
    cluster_matrix = linkage(matrix, method)  # 得到聚类后的链接矩阵
    clusters = fcluster(cluster_matrix, t, criterion)  # 得到聚类结果 (打印出来就是平面簇编号)
    return cluster_matrix, clusters


def show_hierarchical_cluster(cluster_matrix, figsize):
    plt.figure(figsize=figsize)  # 设置画布大小
    dendrogram(cluster_matrix)  # 绘制树状图，通常用于可视化层次聚类结果
    plt.show()  # 绘制工作在缓存中进行，只有调用plt.show()，才会将缓存的绘制结果在画布显示出来，


def density_clustering(tensor, eps, min_samples):
    """
    密度聚类方法
    :param tensor:
    :param eps: 定义了样本之间的最大距离阈值
    :param min_samples: 表示核心点创建邻域需要的最少样本数
    :return:
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(tensor)  # fit方法进行真正的聚类
    return tensor, dbscan.labels_   # 聚类结果


def show_density_cluster(scatters, colors, markers, labels):
    for i in range(len(scatters)):
        plt.scatter(scatters[i][:, 0], scatters[i][:, 1], c=colors[i], marker=markers[i], label=labels[i])
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend(loc=2)
    plt.show()
