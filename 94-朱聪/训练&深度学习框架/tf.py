import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 通过tf搭建神经网络

'''
搭建神经网络，需要 输入数据，隐藏层，输出层，之间的权重，激活函数，损失函数(损失值)，反向传播算法，学习率
'''

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis] # 效果同reshape(200,1) 将一维数组转为了列向量
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise # y = x^2 + noise

# 定义两个placeholder存放输入数据。占位符，不用立即提供数据。这里定义了输入
x = tf.placeholder(tf.float32, [None, 1]) # 参数：type, shape, name
y = tf.placeholder(tf.float32, [None, 1]) # 定义了一个列向量

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10])) # 1行10列 二维矩阵
biases_L1 = tf.Variable(tf.zeros([1, 10]))  # 加入偏置项
# tf.matmul(A,B) 就是矩阵乘法，同A@B dot(A,B)
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1 # 得到一个 (?, 10)的矩阵
L1 = tf.nn.tanh(Wx_plus_b_L1) # 加入激活函数

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run([Weights_L1, biases_L1]))

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))  # 加入偏置项
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2 # 得到了(?, 1)的列向量
prediction = tf.nn.tanh(Wx_plus_b_L2)  # 加入Tanh激活函数

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 0.1是学习率，.minimize(loss)是最小化loss

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练2000次。不断更新权重和偏置
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值

    # r- 是控制线条样式的参数字符串 r是红色 -是实线   lw=5是线条宽度
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()