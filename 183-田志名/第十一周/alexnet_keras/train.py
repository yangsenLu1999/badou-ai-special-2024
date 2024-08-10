from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adam
from model import AlexNet
import numpy as np
import utils
import cv2
import tensorflow.compat.v1 as tf
import os

from keras import backend as K
#K.set_image_dim_ordering('tf')
tf.compat.v1.enable_eager_execution()
'''
K.set_image_dim_ordering('tf') 的作用是将 Keras 的图像维度顺序设置为与 TensorFlow 框架一致的顺序。
在这种顺序下，图像数据的维度顺序为：(batch, height, width, channels)。
'''

#model.fit_generator要求输入是一个迭代器
def data_loader(data,batch_size):
    np.random.shuffle(data)
    n=len(data)
    i=0
    while 1:
        images = []
        labels = []
        for j in range(batch_size):
            img=cv2.imread(r"./train/"+data[i].split(";")[0])      #加了r之后\就不是转义字符了
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
            label=data[i].split(";")[1]
            images.append(img)
            labels.append(label)
            i = (i+1) % n
        X=utils.resize_image(images,(224,224))
        X=X.reshape(-1,224,224,3)
        Y=to_categorical(labels)
        yield (X,Y)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if tf.test.is_gpu_available():
        print("GPU is available")
    else:
        print("GPU is not available")
    log_dir = "./logs/"        #模型保存的位置
    batch_size=128
    with open("dataset.txt","r") as f:
        dataset=f.readlines()
    np.random.seed(10101)
    np.random.shuffle(dataset)

    # 90%用于训练，10%用于估计。
    num_val = int(len(dataset)*0.1)
    num_train = len(dataset) - num_val

    model = AlexNet()

    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_accuracy',
                                    save_weights_only=False,
                                    save_best_only=True,
                                    period=1
                                )
    '''
    ModelCheckpoint是Keras中的一个回调函数，用于在训练过程中保存模型的权重。它可以帮助我们在训练过程中定期保存模型的状态，
    以便在训练中断后可以从上次保存的点继续训练，或者在训练完成后保留最佳模型。
    filepath: 保存模型权重的文件路径，可以包含文件名和扩展名。例如'model_weights.h5'。
    monitor: 需要监视的指标，例如'val_loss'表示监视验证集的损失。acc精度
    save_best_only: 如果设置为True，则只保存在验证集上性能最好的模型权重；如果设置为False，则会在每个epoch结束时都保存模型权重。
    period：每隔多少个epoch保存一下模型
    verbose: 是否打印保存信息
    '''
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_accuracy',
                            factor=0.5,
                            patience=3,
                            verbose=1
                        )
    '''
    ReduceLROnPlateau是PyTorch中的一个学习率调整策略，它用于在训练过程中自动降低学习率。
    monitor: 需要监视的指标，例如'val_loss'表示监视验证集的损失。acc则代表精度
    factor: 学习率降低的倍数，例如0.5表示将学习率乘以0.5。
    patience: 当指标连续patience个epoch没有改善时，学习率将被降低。
    min_lr: 学习率的最小值，防止学习率过低。
    verbose: 是否打印学习率调整信息。=1打印
    '''
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=1
                        )
    '''
    EarlyStopping是Keras中的一个回调函数，用于在训练过程中提前停止训练。
    通过使用EarlyStopping回调函数，我们可以监控验证集的性能指标（如损失），并在连续多个epochs没有改善时停止训练。这样可以避免模型过度拟合训练数据，从而降低过拟合的风险。
    monitor: 需要监视的指标，例如'val_loss'表示监视验证集的损失。
    patience: 当指标连续patience个epoch没有改善时，停止训练。
    mode: 'min'表示当指标停止下降时停止训练，'max'表示当指标停止上升时停止训练。
    verbose: 是否打印停止信息。
    '''

    # 交叉熵
    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr=1e-3),
            metrics = ['accuracy'])

    model.fit_generator(data_loader(dataset[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_loader(dataset[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=20,
            initial_epoch=0,
            callbacks=[checkpoint_period1, reduce_lr])
    model.save_weights(log_dir+'last1.h5')

    '''
    model.fit_generator是Keras早期版本中的一个方法，用于处理数据生成器（generator）作为输入。在Keras 2.4之后，这个方法已经被弃用，建议使用model.fit方法，它可以接受任何可迭代的数据生成器作为输入。
    model.fit方法可以直接接受NumPy数组、TensorFlow张量或tf.data.Dataset对象作为输入，而不需要使用数据生成器。这使得它在处理大规模数据集时更加方便。
    model.fit_generator需要指定steps_per_epoch参数，表示每个训练周期需要遍历多少批次数据。而model.fit会自动计算这个值，无需手动设置。
    model.fit_generator还可以接受一个名为validation_data的参数，用于指定验证集。而model.fit则通过validation_split参数来指定验证集的比例。
    model.fit_generator可以同时处理多个输入和输出，而model.fit只能处理单个输入和输出。
    '''