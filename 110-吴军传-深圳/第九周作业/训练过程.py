"""
训练过程代码，使用梯度下降法：
1、生成数据集
2、定义模型和损失函数
3、计算梯度并更新参数
4、训练模型
"""
import numpy as np
import matplotlib.pyplot as plt

#生成数据集
np.random.seed(24)
x = 2*np.random.rand(100,1)
y = 4+3 * x + np.random.randn(100,1)

#设置学习率和迭代次数
learning_rate = 0.1
n_iteratinons = 1000
m = 100

#初始化参数
theta = np.random.randn(2,1)

#添加偏置项
X_b = np.c_[np.ones((100,1)),x]

#损失函数
def compute_loss(x,y,theta):
    predictions=x.dot(theta)
    errors = predictions-y
    return (1/(2 * m)*np.sum(errors**2))

#梯度下降
for iteration in range(n_iteratinons):
    gradients = 1/m * X_b.T.dot(X_b.dot(theta)-y)
    theta = theta-learning_rate*gradients

    #打印损失值
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {compute_loss(X_b, y, theta)}")
print(f"Final parameters: {theta}")

#绘制结果
plt.scatter(x,y)
plt.plot(x,X_b.dot(theta),color='red')
plt.xlabel("x")
plt.ylabel('y')
plt.title("Linear Regression Fit")
plt.show()



