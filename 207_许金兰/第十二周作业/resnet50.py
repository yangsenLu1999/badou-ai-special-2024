import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


class ResNet50:
    def __init__(self, input_shape, classes=1000):
        """
        参数初始化
        :param input_shape: 输入数据维度
        :param classes:分类类目数
        """
        self.input_shape = input_shape
        self.classes = classes

    def conv_block(self, inputs, kernel_size, filters, stage, block, strides=(2, 2)):
        """
        conv_block
        :param inputs:输入数据
        :param kernel_size:卷积核大小
        :param filters:卷积输出空间的维数
        :param stage:
        :param block:
        :param strides:
        :return:
        """
        filters1, filters2, filters3 = filters

        name_base_conv = 'res' + str(stage) + block + '_branch'
        name_base_bn = 'bn' + str(stage) + block + '_branch'
        # Conv2D参数解析Conv2D(filter,kernel_size)
        # filters：整数，表示输出空间的维度（即卷积核的数量）。它决定了卷积层的输出通道数。
        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=name_base_conv + '2a')(inputs)  # 进行了数据降维，降到64个通道
        x = BatchNormalization(name=name_base_bn + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   name=name_base_conv + '2b')(x)
        x = BatchNormalization(name=name_base_bn + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=name_base_conv + '2c')(x)  # 升维
        x = BatchNormalization(name=name_base_bn + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=name_base_conv + '1')(inputs)
        shortcut = BatchNormalization(name=name_base_bn + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def identity_block(self, inputs, kernel_size, filters, stage, block):
        """
        identity_block
        :param input_tensor: 输入数据
        :param kernel_size: 卷积核大小
        :param filters:卷积输出维数
        :param stage:
        :param block:
        :return:
        """
        filters1, filters2, filters3 = filters
        name_base_conv = 'res' + str(stage) + block + '_branch'
        name_base_bn = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=name_base_conv + '2a')(inputs)
        x = BatchNormalization(name=name_base_bn + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same', name=name_base_conv + '2b')(x)
        x = BatchNormalization(name=name_base_bn + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=name_base_conv + '2c')(x)
        x = BatchNormalization(name=name_base_bn + '2c')(x)
        # 与输入相加
        x = layers.add([x, inputs])
        x = Activation('relu')(x)
        return x

    def net(self):
        input_img = Input(shape=self.input_shape)  # 构建网络的第一层：输入层
        x = ZeroPadding2D((3, 3))(input_img)  # 边缘填充
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)

        x = Flatten()(x)  # 拉平
        x = Dense(self.classes, activation='softmax', name='fc1000')(x)  # 全连接层，units=classes输出维度
        # 加载模型参数
        model = Model(inputs=input_img, outputs=x, name='resnet50')
        model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
        return model


if __name__ == '__main__':
    model = ResNet50([224, 224, 3]).net()
    model.summary()
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    print('Predicted:', decode_predictions(predictions))
