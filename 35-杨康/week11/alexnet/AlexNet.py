from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization
from keras.models import Sequential

def AlexNet(input_shape=(224,224,3),output_shape=2):
    model = Sequential()
    #1.使用大小为11*11，步长为4*4的96个卷积核，输出为（55，55，96）
    # 为学习效率这里使用48个卷积核
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(11,11),
            strides=(4,4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )

    model.add(BatchNormalization())
    #2.使用大小为（3，3），步长为（2，2）的最大池化层，输出为（27，27，96）
    model.add(
        MaxPool2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )

    )
    # 3.使用大小为5*5，步长为1*1的256个卷积核，输出为（27，27，256）
    # 为学习效率这里使用128个卷积核
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    model.add(BatchNormalization())
    # 4.使用大小为（3，3），步长为（2，2）的最大池化层，输出为（13，13，256）
    model.add(
        MaxPool2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    # 5.使用大小为3*3，步长为1*1的384个卷积核，输出为（13，13，384）
    # 为学习效率这里使用192个卷积核
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 6.使用大小为3*3，步长为1*1的384个卷积核，输出为（13，13，384）
    # 为学习效率这里使用192个卷积核
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 7.使用大小为3*3，步长为1*1的256个卷积核，输出为（13，13，256）
    # 为学习效率这里使用128个卷积核
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 8.使用大小为（3，3），步长为（2，2）的最大池化层，输出为（6，6，256）
    model.add(
        MaxPool2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    #9.使用两个4096全连接层，最后输出1000，这里是输出2
    #缩减为1024
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_shape,activation='softmax'))
    return model


