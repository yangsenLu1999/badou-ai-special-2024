import numpy as np
from model.AlexNet import AlexNet
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import cv2
import utils
import keras.utils as np_utils


def generate_arrays_from_file(lines, batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r".\data\image\train" + '/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i + 1) % n
        # 处理图像
        X_train = utils.resize_image(X_train, (224, 224))
        X_train = X_train.reshape(-1, 224, 224, 3)
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
        yield (X_train, Y_train)


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = 'logs/'

    # 打开数据集的txt
    with open(r"data/dataset.txt", "r") as f:
        lines = f.readlines()

    # 打乱行，用于读取数据训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于预测
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 建立模型
    model = AlexNet()

    # 每3代保存一次
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_los:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )

    # 准确率三次不下降就降低学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )

    # 当val_loss一直不下降代表模型训练基本完成，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 交叉熵
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    # 一次训练集的大小
    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr])
    model.save_weights(log_dir + 'last1.h5')
