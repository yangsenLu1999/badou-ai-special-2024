import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 你的绘图代码
# plt.plot(l, cs)
# plt.plot(z, cs)
# plt.show()

# 几种标准化方法

def Normalization1(x):
    '''归一化（0~1）,x_=(x−x_min)/(x_max−x_min)'''
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


def Normalization2(x):
    '''归一化（-1~1）,x_=(x−x_mean)/(x_max−x_min)'''
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]


def z_score(x):
    '''x∗=(x−μ)/σ，μ 是均值（mean），σ 是标准差（standard deviation）'''
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    '''这行代码计算 x 的方差。它使用列表推导式计算每个值与均值的差的平方，
    然后对所有这些平方值求和，最后除以元素个数 len(x) 以得到方差。
    注意：这样计算的是样本方差（实际上是二次方的均值）'''
    return [(i - x_mean) / s2 for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1 = []

'''计算一个列表 l 中每个元素出现的次数，并将结果存储在一个新的列表 cs 中'''
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)

n1 = Normalization1(l)
n2 = Normalization2(l)
z = z_score(l)
print(n1)
print(n2)
print(z)

# 绘制所有的数据
plt.plot(l, cs,  label='原始数据', color='blue')
plt.plot(n1, cs, label='Normalization1', color='orange')
plt.plot(n2, cs, label='Normalization2', color='green')
plt.plot(z, cs, label='Z-score', color='red')

# 添加图例
plt.legend()

# 添加标题和标签
plt.title('数据归一化比较')
plt.xlabel('样本索引')
plt.ylabel('值')

# 显示图形
plt.show()
