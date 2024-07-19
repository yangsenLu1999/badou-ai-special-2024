from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.alexnet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K
#K.set_image_dim_ordering('tf')

def generate_arrays_from_file(lines,batch_size):
    n=len(lines)
    i=0
    while 1:
        x_train=[]
        y_train=[]
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name=lines[i].split(";")[0]
            img=cv2.imread(r".\data\train" + '/' + name)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=img/255

            x_train.append(img)
            y_train.append(lines[i].split(";")[1])

            i=(i+1) % n
        x_train=utils.resize_image(x_train,(224,224))
        x_train=x_train.reshape(-1,224,224,3)
        y_train=np_utils.to_categorical(np.array(y_train),num_classes= 2)
        yield (x_train,y_train)

if __name__=="__main__":
    # 保存模型
    log_dir = "./logs/"

    # 打开数据集
    with open(r".\data\dataset.txt",'r') as f:
        lines=f.readlines()

    # 打乱数据集的图片顺序，训练出更好的模型
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 分配训练和推理的照片数量
    number_valuate=int(len(lines)) * 0.1
    number_train=int(len(lines) - number_valuate)
    print(number_train)

    # 导入模型
    model=AlexNet()

    # 每训练3次保存一下模型
    checkpoint_period1=ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor="acc",
        save_weights_only=False,
        save_best_only=True,
        period=3
    )

    # 降低学习率
    reduce_lr=ReduceLROnPlateau(
        monitor="acc",
        factor=0.5,
        patience=3,
        verbose=1
    )

    # 是否需要早停，防止过拟合
    early_stopping=EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=1
    )

    #交叉熵
    model.compile(
        loss= 'categorical_crossentropy',
        optimizer=Adam(lr=1e-3),
        metrics=['accuracy']
    )

    # 设置batch——size大小
    batch_size=128
    # print('Train on {} samples, val on {} samples, with batch size {}.'.format(number_train, number_valuate, batch_size))
    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:number_train], batch_size),
                        steps_per_epoch=max(1, number_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[number_train:], batch_size),
                        validation_steps=max(1, number_valuate // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr])
    model.save_weights(log_dir + 'last1.h5')





