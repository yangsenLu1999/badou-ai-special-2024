import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import AlexnetModel
import AlexnetUtils
from keras.utils import to_categorical
from keras.utils import np_utils

import keras.backend as K
# K.set_image_dim_ordering('tf')
# K.image_data_format() == 'channels_first'


def input_preprogress(lines, batch_size):  # 按照batchsize取出数据
    n = len(lines)  # 2500 + 22500
    i = 0
    print(lines[0])
    while 1:
        image_train = []
        label_train = []
        for j in range(batch_size):  # 每次获取一个batch_size大小的数据加到空列表中
            if i == 0:
                np.random.shuffle(lines)  # 不返回任何值，所以不能用lines = np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # print(lines[i].split[';'][1])
            # img = cv2.imread('data/image/train/' + name)
            img = cv2.imread(r'.\data\image\train' + '/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            image_train.append(img)
            label_train.append(lines[i].split(';')[1])
            i = (i+1) % n  # 读完一个周期后重新开始，如果用i += 1，会导致i不断累加，超过lines的边界
        image_train = AlexnetUtils.resize_img(image_train, (224, 224))
        image_train = image_train.reshape(-1, 224, 224, 3)
        label_train = np_utils.to_categorical(np.array(label_train), num_classes=2)  # label_train = np.array(label_train)
        yield (image_train, label_train)
    pass


if __name__ == '__main__':
    log_dir = 'logs/'  # 预留模型保存的位置
    with open(r'.\data\dataset.txt', 'r') as f:  # utils根据图片集，录写了图片名称和标签tet，打开并读取
        lines = f.readlines()
        # print(lines)  # 'dog.9924.jpg;0\n', 'dog.9925.jpg;0\n',.......
        # print(len(lines))  # 25000
    np.random.seed(10101)
    # 设置的随机种子，代表按照固定随机规律打乱顺序，即每次重新运行代码lines打乱后的顺序一样
    np.random.shuffle(lines)  # 打乱数据更有利于训练
    # print(lines)
    np.random.seed(None)
    num_train = int(0.9 * len(lines))  # 列表的长度用len读取，90%用于训练，10%用于测试，alexnettrain中fitgenerator对训练集和验证集都进行了计算
    num_test = len(lines) - num_train
    # print(num_train, num_test)
    # pytorch做mnist是先定义net网络结构，再定义model包含训练推理反向更新等过程；
    # alexnet也先定义net网络结构，但keras接口在形成net的时候一并组建了模型，调用时(net+model)封装在一起。参照keras接口做mnist识别
    model = AlexnetModel.Alexnet()  # 调用alexnet网络(包括了模型结构的训练推理过程fit, evaluate, predict等等)

    checkpoint = ModelCheckpoint(filepath=log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='accuracy', save_weights_only=False, save_best_only=True, period=3)
    '''
    以指定时间间隔保存模型或权重为 checkpoint 文件，以便稍后可以加载模型或权重，从而从保存的状态继续训练.
    filepath: 保存模型文件的路径，string 
    monitor: 要监控指标的名称。指标一般通过 Model.compile 方法设置
    verbose: 详细模式，0: silent; 1 在 callback 执行时显示消息
    save_weights_only: True 表示只保存模型的权重 model.save_weights(filepath)，否则保存整个模型 model.save(filepath)。
    save_best_only: True 只在模型被认为是目前最好时保存
    '''
    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1)  # acc三次不下降就下降学习率继续训练
    '''
    keras中的回调函数ReduceLROnPlateau优化器：定义学习率之后，经过一定epoch迭代之后，模型效果不再提升，该学习率可能已经不再适应该模型
    factor：学习率衰减的因子，触发条件后lr*=factor；
    patience：不再减小（或增大）的累计次数；在多少个epoch内验证集指标没有改善时才进行学习率调整
    '''
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    # 这里定义了用于验证集的早停方法，本模型没有进行调用
    # val_loss是验证集的损失，早停是依照验证集的损失来做判断，训练集损失用于早停没意义，一直不下降的时候意味着模型基本训练完毕

    # 和keras接口识别mnist一样，model定义好以后可以用model.compile编译，model.fit训练，model.evaluate测试，model.predict推理
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])  # 优化器SGD, RMSP, ADAM；损失函数交叉熵
    batch_size = 150
    print('train examples: {}, test examples: {}, batch_size: {}'.format(num_train, num_test, batch_size))

    # fit_generate更适用于自己写函数每次读一个batchsize的图像训练这种情况，fit是直接传入全部训练集, 一次batch占内存会少一些, 从numtrain中不断取出batchsize个数据进行训练

    # print(input_preprogress(lines[:num_train], batch_size))
    model.fit_generator(input_preprogress(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train//batch_size),
                        validation_data=input_preprogress(lines[num_train:], batch_size),
                        validation_steps=max(1, num_test//batch_size),
                        epochs=1,
                        initial_epoch=0,
                        callbacks=[checkpoint, reduce_lr])
    # 这里其实对全部数据[9成训练+1成验证],都进行了fit
    model.save_weights(log_dir + 'last1.h5')

