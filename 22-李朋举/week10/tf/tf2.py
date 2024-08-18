import tensorflow as tf
import numpy as np

'''
[tensorflow  超参数调整、数据预处理、模型评估等步骤、管理训练过程和结果]
1. 生成了带有噪声的平方函数数据作为训练集。
2. 定义了一个简单的神经网络模型，包括两层和激活函数。
3. 定义了损失函数和优化器，使用均方误差作为损失函数，梯度下降作为优化器。
4. 创建了 TensorFlow 会话，并进行了变量初始化。
5. 进行了超参数调整，包括训练轮数和批量大小。
6. 对数据进行了预处理，使用 `tf.data.Dataset` 进行数据加载和打乱，并设置了批量大小。
7. 在训练循环中，使用批量数据进行训练，并更新模型参数。
8. 同时，使用 `MeanAbsoluteError` 指标评估模型在每个轮次上的性能。
9. 训练完成后，保存了模型，并可以进行恢复。
10. 最后，使用 matplotlib 可视化了真实数据和预测数据的对比。
'''
# 生成数据
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise


# 定义神经网络模型
def neural_network(x):
    # 第一层
    weights_L1 = tf.Variable(tf.random_normal([1, 10]))
    biases_L1 = tf.Variable(tf.zeros([1, 10]))
    Wx_plus_b_L1 = tf.matmul(x, weights_L1) + biases_L1
    L1 = tf.nn.tanh(Wx_plus_b_L1)

    # 输出层
    weights_L2 = tf.Variable(tf.random_normal([10, 1]))
    biases_L2 = tf.Variable(tf.zeros([1, 1]))
    Wx_plus_b_L2 = tf.matmul(L1, weights_L2) + biases_L2
    prediction = tf.nn.tanh(Wx_plus_b_L2)

    return prediction


# 定义损失函数和优化器
def loss_function(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


optimizer = tf.train.GradientDescentOptimizer(0.1)

# 定义训练步骤
train_step = optimizer.minimize(loss_function)

# 创建 TensorFlow 会话
with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())

    # 超参数调整
    num_epochs = 2000
    batch_size = 32

    # 数据预处理
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)

    # 模型评估指标
    mae = tf.keras.metrics.MeanAbsoluteError()

    # 训练循环
    for epoch in range(num_epochs):
        for batch_x, batch_y in dataset:
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})

        # 模型评估
        mae_value = mae.update_state(batch_y, sess.run(neural_network(batch_x), feed_dict={x: batch_x}))
        print(f"Epoch {epoch}: MAE = {mae_value.numpy()}")

    # 保存模型
    saver = tf.train.Saver()
    save_path = saver.save(sess, "model.ckpt")
    print("Model saved at:", save_path)

    # 恢复模型
    saver.restore(sess, "model.ckpt")
    print("Model restored")

    # 可视化结果
    import matplotlib.pyplot as plt

    plt.scatter(x_data, y_data, label="True Data")
    plt.plot(x_data, sess.run(neural_network(x_data), feed_dict={x: x_data}), color="red", label="Predicted Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Neural Network Prediction")
    plt.legend()
    plt.show()



