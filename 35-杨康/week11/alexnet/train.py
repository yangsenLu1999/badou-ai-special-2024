from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
import utils
from AlexNet import AlexNet
import numpy as np
import cv2

def generate_arrays_from_file(lines,batch_size):
    n = len(lines)
    i = 0
    while 1:
        x_train = []
        y_trian = []
    #获取一个batchsize大小的数据
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            img = cv2.imread('./data/image/train/'+name)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_nor = img_rgb/255
            x_train.append(img_nor)
            y_trian.append(lines[i].split(';')[1])
            #处理完一个周期后重新开始
            i = (i+1) % n
        x_train = utils.resize_image(x_train,(224,224))
        x_train = x_train.reshape(-1,224,224,3)
        y_trian = np_utils.to_categorical(y_trian,num_classes=2)
        yield (x_train, y_trian)


if __name__ == '__main__':
    #保存位置
    log_dir = './logs/'
    #读取数据集
    with open('./data/dataset.txt') as f:
        lines = f.readlines()
    #打乱数据
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    # 90%用于训练，10%用于测试
    num_test = int(len(lines)*0.1)
    num_train = len(lines) - num_test
    #建立AlexNet模型
    model = AlexNet()
    # 保存的方式，3世代保存一次
    checkpoint_period = ModelCheckpoint(
                                log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                monitor='accuracy',
                                save_weights_only=False,
                                save_best_only=True,
                                period=3
                                 )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                                monitor='accuracy',
                                patience=3,
                                factor=0.5,
                                verbose=1
                                 )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                                monitor='val_loss',
                                min_delta=0,
                                patience=10,
                                verbose=1
                                )
    #编译模型
    model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
            )
    #bacth大小
    batch_size = 128
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_test, batch_size))
    #开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_test // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period, reduce_lr])
    model.save_weights(log_dir + 'last1.h5')