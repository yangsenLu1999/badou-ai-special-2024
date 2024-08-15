# 一个完整的tensorflow 的训练过程，可跟keras做对比，keras相当于在这个上面做了个封装
# 用tf完整的写了一个神经网络的训练过程，比纯手写neuralnetwork简单很多，比keras又复杂
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]      # 随机生成一些Input值，也可以换成自己的数据集
'''np.linspace(start, stop, num)：生成从 start 到 stop 的 num 个均匀分布的数值。在这里，-0.5 是起始值，0.5 是结束值，200 是生成的数值个数。
[:, np.newaxis]：这个操作将一维数组 x_data 转换为二维数组。原本 x_data 会是形状为 (200,) 的一维数组，使用 np.newaxis 后，它变成了形状为 (200, 1) 的二维数组'''
noise = np.random.normal(0, 0.02, x_data.shape)
'''np.random.normal(loc, scale, size)：生成符合正态分布（高斯分布）的随机数。
loc 是分布的均值，这里是 0。
scale 是分布的标准差，这里是 0.02。
size 是输出数组的形状，这里使用 x_data.shape 使得生成的噪声与 x_data 的形状一致。'''
y_data = np.square(x_data) + noise                       # 加点噪声，square一下变成正确答案，标签，随便设的
'''np.square(x_data)：计算 x_data 中每个元素的平方'''

# 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32, [None, 1])                # 先占位置，这样就能在上面改了
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))      # 随机生成一个1-10之间的w
biases_L1 = tf.Variable(tf.zeros([1, 10]))               # 加入偏置项,随机生成
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1      # 对x做一个wx+b
L1 = tf.nn.tanh(Wx_plus_b_L1)                            # 加入激活函数tanh，得到隐藏层的输出

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))                # 加入偏置项
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)                    # 加入激活函数，得到最终输出值

# 定义损失函数（均方差函数）MSE
loss = tf.reduce_mean(tf.square(y - prediction))         # 这里没调接口，手写的mse公式
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)     # 这里很简洁，梯度下降法接口，步长0.1

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())          # 要先run变量的init op
    # 训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})      # feed填入占位的

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()

