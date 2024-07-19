import tensorflow.compat.v1 as tf
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Softmax,BatchNormalization,Dropout




tf.disable_v2_behavior()
slim = tf.contrib.slim

def vgg_16(inputs,num_classes=1000,
           is_training = True,
           dropout_keep_prob =0.5,
           spatial_squeeze = True,
           scope = 'vgg_16'
           ):
    with tf.variable_scope(scope,'vgg_16',[inputs]):

        model = Sequential()

        model.add(
            Conv2D(
                filters=64,
                kernel_size=(3,3),
                strides=(1,1),
                padding='valid',
                input_shape= inputs,
                activation='relu'
            )
        )
        model.add(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='valid',
                input_shape=inputs,
                activation='relu'
            )
        )
        model.add(BatchNormalization())
        model.add(
            MaxPooling2D(
            padding=(2,2),
            strides=(2,2),

        ))



    # 第二次【3，3】 卷积网络，输出特征层是128
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            input_shape=inputs,
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            input_shape=inputs,
            activation='relu'
        )
    )
    model.add(BatchNormalization())
    model.add(
        MaxPooling2D(
            padding=(2, 2),
            strides=(2, 2),
        ))

    # 第三次 【3，3】卷积网络  28 28 512
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            input_shape=inputs,
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            input_shape=inputs,
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            input_shape=inputs,
            activation='relu'
        )
    )
    model.add(BatchNormalization())
    model.add(
        MaxPooling2D(
            padding=(2, 2),
            strides=(2, 2),

        ))


    # 第四次  输入是14 ， 14 512
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            input_shape=inputs,
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            input_shape=inputs,
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            input_shape=inputs,
            activation='relu'
        )
    )
    model.add(BatchNormalization())
    model.add(
        MaxPooling2D(
            padding=(2, 2),
            strides=(2, 2),

        ))

    # 进入全连接层 利用卷积模拟全连接 输出net（1，1，4096）

    model.add(
        Conv2D(
            filters=4096,
            kernel_size=(7,7),
            strides=(1,1),
            padding='valid'
        )
    )

    model.add(Dropout(
        rate=dropout_keep_prob,
        is_training=True,

    ))




