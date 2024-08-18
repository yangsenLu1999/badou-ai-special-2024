# -*- coding:utf-8 -*-

__author__ = 'Young'

import cv2

import ClusterUtil as CUtil
from sklearn import datasets
import SIFTUtil

"""
【第七周作业】
1.实现层次聚类 2.实现密度聚类 3.实现SIFT
"""

"""
1.实现层次聚类
"""
original_data = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
single_h_cluster = CUtil.hierarchical_clustering(original_data, 'single', 4, 'distance')
single_h_maxclust_cluster = CUtil.hierarchical_clustering(original_data, 'single', 4, 'maxclust')
weighted_h_cluster = CUtil.hierarchical_clustering(original_data, 'weighted', 4, 'distance')
ward_h_cluster = CUtil.hierarchical_clustering(original_data, 'ward', 4, 'distance')
CUtil.show_hierarchical_cluster(single_h_cluster[0], (5, 3))
CUtil.show_hierarchical_cluster(single_h_maxclust_cluster[0], (5, 3))
CUtil.show_hierarchical_cluster(weighted_h_cluster[0], (5, 3))
CUtil.show_hierarchical_cluster(ward_h_cluster[0], (5, 3))
print('single cluster \n', single_h_cluster[1])
print('single maxclust \n', single_h_maxclust_cluster[1])
print('weighted cluster \n', weighted_h_cluster[1])
print('weighted cluster \n', ward_h_cluster[1])

"""
2.实现密度聚类
"""
iris = datasets.load_iris()
tensor_data = iris.data[:, :4]  # 只取特征空间中的4个维度
CUtil.show_density_cluster([tensor_data], ['red'], ['o'], ['see'])
density_cluster = CUtil.density_clustering(tensor_data, 0.4, 9)
scatters = [density_cluster[0][density_cluster[1] == 0], density_cluster[0][density_cluster[1] == 1],
            density_cluster[0][density_cluster[1] == 2]]
colors = ['red', 'green', 'blue']
markers = ['o', '*', '+']
labels = ['label0', 'label1', 'label2']
CUtil.show_density_cluster(scatters, colors, markers, labels)

"""
3.实现SIFT
"""
lenna_key_points = SIFTUtil.key_points_image('../Image/lenna.png')
cv2.imshow('lenna sift key points', lenna_key_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

SIFTUtil.draw_matchs_knn('../Image/iphone1.png', '../Image/iphone2.png')
