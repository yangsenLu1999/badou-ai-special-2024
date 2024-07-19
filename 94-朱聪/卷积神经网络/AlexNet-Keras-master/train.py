from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K
K.set_image_dim_ordering('tf')

def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r".\data\image\train" + '/' + name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img/255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = utils.resize_image(X_train,(224,224))
        X_train = X_train.reshape(-1,224,224,3)
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 2)   
        yield (X_train, Y_train)


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r".\data\dataset.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # 建立AlexNet模型
    model = AlexNet()

    # 保存的方式，3世代(每3个epoch)保存一次
    # 回调函数定义

    checkpoint_period1 = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='acc', # 监控模型的精度
                                    save_weights_only=False, # 保存整个模型（包括结构和权重），而不仅仅是权重
                                    save_best_only=True, # 只保存在验证集上性能最好的模型
                                    period=3 # 每3个epoch保存一次
                                )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='acc',
                            factor=0.5, # 减少学习率的因子，新的学习率将是当前学习率乘以此因子
                            patience=3, # 如果连续3个epoch监控的指标没有改善，则执行学习率调整操作
                            verbose=1 # 显示学习率调整的信息
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss', # 监控验证集上的损失
                            min_delta=0, # 验证集上的损失的改善的最小变化，小于这个值被认为没有改善
                            patience=10, # 如果连续10个epoch验证集损失没有改善，则停止训练
                            verbose=1 # 显示早停信息
                        )

    # 交叉熵
    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr=1e-3),
            metrics = ['accuracy'])

    # 一次的训练集大小
    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size), # 生成器函数，用于生成训练和验证数据批次
            steps_per_epoch=max(1, num_train//batch_size), # 每轮【训练】的步数，即每个epoch遍历的样本数
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size), # 每轮【验证】的步数，即每个epoch遍历的样本数
            epochs=50, # 训练的轮数
            initial_epoch=0, # 初始的轮数
            callbacks=[checkpoint_period1, reduce_lr, early_stopping]) # 传入之前定义的回调函数列表

    model.save_weights(log_dir+'last1.h5') # 保存最终训练完成的模型权重

