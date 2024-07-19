import tensorflow as tf
from tensorflow import keras # # keras是TF的高级API，用起来更加的方便，一般也是用keras。
from tensorflow.keras import layers
import numpy as np
import gzip

# TF实现Fashion MNIST时尚服装分类， 也是一个十分类任务

# fashion_mnist = keras.datasets.fashion_mnist   # keras.datasets中也封装了一些常用的数据集
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 这里下载下来了
def load_data():
    files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    with gzip.open(files[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(files[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(files[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(files[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

(train_images, train_labels), (test_images, test_labels) = load_data()
print('train_images shape:',train_images.shape) # (60000, 28, 28)
print('train_labels shape:',train_labels.shape) # (60000,)
print('test_images shape:',test_images.shape) # (10000, 28, 28)
print('test_labels shape:',test_labels.shape) # (10000,)

# 标准化 归一化
train_images = train_images / 255.0 * 0.99 + 0.01 # 防止除255之后过小为0，控制在0.01 ~ 1之间
test_images = test_images / 255.0 * 0.99 + 0.01

# 构建网络
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)), # 展开成一个维度，因为是全连接层
    # PyTorch的全连接层需要一个输入神经元数量和输出数量torch.nn.Linear(5,10)，且就是代表线性变换层
    # 而keras中的Dense是【不需要输入参数】的keras.layers.Dense(10)
    layers.Dense(128, activation='relu'), # Dense还可以包括 激活函数，初始化也可以直接在构造函数中完成
    layers.Dense(10, activation='softmax')
])

# 优化器，损失函数和优化器还有metric衡量指标 都在模型的编译函数中设置完成
# adam优化器，损失函数交叉熵，衡量模型性能用了accuracy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练集
model.fit(train_images, train_labels, epochs=10) # 非常简洁

# 验证集
test_loss, test_acc = model.evaluate(test_images, test_labels)

# 测试集，进行未知标签样本的类别推理的
res = model.predict(test_images)

# 前500张图片的情况
for i in range(500):
    print('该图片实际分类是', test_labels[i])
    if np.argmax(res[i]) != test_labels[i]:
        print('incorrect')
    print('该图片预测分类是', np.argmax(res[i]))
