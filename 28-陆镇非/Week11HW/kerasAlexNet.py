from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import random


def AlexNet(input_shape=(32, 32, 3), output_shape=10):
    # AlexNet
    model = Sequential()

    # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
    # 所建模型后输出为48特征层
    # ((h-kernel_h)/stride_h) + 1
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            input_shape=input_shape,
            activation='relu'
        )
    )

    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )
    )
    # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
    # 所建模型后输出为128特征层
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )
    )
    # batchSize*7*7*128
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)；
    # 所建模型后输出为128特征层
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256)；
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    # 两个全连接层，最后输出为1000类,这里改为2类
    # 缩减为1024
    # batchSize*3*3*128
    model.add(Flatten())
    model.add(Dense(1152, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape, activation=None))

    return model


# 加载cifar-10数据集
def dataLoad():
    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.cifar10.load_data()
    # 数据集进行归一化
    train_data = train_data / 255
    test_data = test_data / 255
    # # 将标签数据集从数组类型array修改成整形类型int
    # train_label.astype(np.int)
    # test_label.astype(np.int)
    # train_data = tf.constant(train_data, dtype=tf.float64)
    # train_label = tf.constant(train_label, dtype=tf.int32)
    # test_data = tf.constant(test_data, dtype=tf.float64)
    # test_label = tf.constant(test_label, dtype=tf.int32)
    return train_data, train_label, test_data, test_label


class DataIter(object):
    def __init__(self, X, Y, batch_size):
        self.features = X
        self.labels = Y
        self.batchSize = batch_size

    def data_iter(self):
        num_examples = self.features.shape[0]
        indices = list(range(num_examples))
        while(1): # tensorflow data_iter need while(1), different from pytorch training process
            random.shuffle(indices)
            for i in range(0, num_examples, self.batchSize):
                batch_indices = (indices[i: min(i + self.batchSize, num_examples)])
                # print(self.features[batch_indices].shape)
                # print(self.labels[batch_indices].shape)
                yield self.features[batch_indices], self.labels[batch_indices].reshape(-1)


def lossFunc(y, y_predict):
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # automaticly transfer index to one-hot, and it has already had softmax
    loss = loss_func(y, y_predict)
    return loss


if __name__ == "__main__":
    train_data, train_label, test_data, test_label = dataLoad()
    batch_size = 128
    trainingDataSet = DataIter(train_data, train_label, batch_size)
    validationDataSet = DataIter(test_data, test_label, batch_size)

    num_train = train_data.shape[0]
    num_val = test_data.shape[0]

    model = AlexNet()

    # 模型保存的位置
    log_dir = "./logs/"
    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=False,
        period=1
    )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 交叉熵
    model.compile(loss=lossFunc,
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])


    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    # model.fit(train_data, train_label, epochs=100, batch_size=batch_size)

    # 开始训练
    model.fit_generator(trainingDataSet.data_iter(),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=validationDataSet.data_iter(),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=100,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr])
    model.save_weights(log_dir + 'last_model.h5')



