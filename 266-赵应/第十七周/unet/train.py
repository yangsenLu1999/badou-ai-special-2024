import cv2
import keras.callbacks as callbacks
import numpy as np
from keras.losses import BinaryCrossentropy

from model.unet import get_model
import numpy as np


def train():
    loss_fn = BinaryCrossentropy(from_logits=False)
    batch_size = 1
    # 打开数据集的txt
    with open(r"data/dataset.txt", "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='accuracy', factor=.5, patience=5, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)
    # 每10个迭代保存一次模型
    check_point = callbacks.ModelCheckpoint('./ep{epoch:3d}-loss{loss:.3f}-accuracy-validate-{accuracy:.3f}.h5',
                                            monitor='accuracy', save_weights_only=False, save_best_only=True, period=10)
    model = get_model(1, 1)
    model.summary()
    model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])
    print("开始训练。。。")
    model.fit_generator(generate_arrays_from_file(lines, batch_size),
                        steps_per_epoch=15,
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[early_stop, reduce_lr, check_point])
    model.save_weights("inception_traffic_light.h5")


def generate_arrays_from_file(lines, batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        images = []
        labels = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i]
            # 从文件中读取图像
            img = cv2.imread("data/train/image/" + name.strip(), cv2.IMREAD_GRAYSCALE)
            img_label = cv2.imread("data/train/label/" + name.strip(), cv2.IMREAD_GRAYSCALE)
            images.append(img / 255.0)
            labels.append(img_label / 255.0)
            # 读完一个周期后重新开始
            i = (i + 1) % n
        # 处理图像
        images = np.array(images).reshape(-1, 512, 512, 1)
        labels = np.array(labels).reshape(-1, 512, 512, 1)
        yield (images, labels)


if __name__ == '__main__':
    train()
