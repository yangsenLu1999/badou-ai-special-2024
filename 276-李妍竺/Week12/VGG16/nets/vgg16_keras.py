from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten
from tensorflow.keras.models import Sequential



def vgg_16(input_shape=(224,224,3),
         dropout_keep_prob=0.5,
         ):
    model = Sequential()
    # 建立vgg_16的网络
    # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)
    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',input_shape=input_shape,activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    # 2X2最大池化，输出net为(112,112,64)
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))

    # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    # 2X2最大池化，输出net为(56,56,128)
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='valid'))

    # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    # 2X2最大池化，输出net为(28,28,256)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(28,28,512)
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    # 2X2最大池化，输出net为(14,14,512)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))

    # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(14,14,512)
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    # 2X2最大池化，输出net为(7,7,512)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
    model.add(Conv2D(filters=4096, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Dropout(rate=dropout_keep_prob))
    # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
    model.add(Conv2D(filters=4096, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Dropout(rate=dropout_keep_prob))
    # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
    model.add(Conv2D(filters=1000, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='softmax'))

    # 由于用卷积的方式模拟全连接层，所以输出需要平铺

    model.add(Flatten())

    return model

if __name__ == '__main__':
    model = vgg_16()
    model.summary()