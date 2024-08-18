import numpy as np
from tensorflow import keras


# 定义一个Identity block
def identity_block(inputs, kernel_size, filters, stage, block):
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filter1, (1, 1), name=conv_name_base + '2a')(inputs)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filter2, kernel_size, padding='same',
                            name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2c')(x)
    x = keras.layers.add([x, inputs])
    x = keras.layers.Activation('relu')(x)

    return x


# 定义一个conv_block
def conv_block(inputs, kernel_size, filters, stage, block, strides=(2, 2)):
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filter1, (1, 1),strides=strides,
                            name=conv_name_base + '2a')(inputs)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filter2, kernel_size, padding='same',
                            name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(name=bn_name_base + '2c')(x)

    y = keras.layers.Conv2D(filter3, (1, 1), strides=strides,
                            name=conv_name_base + '1')(inputs)
    y = keras.layers.BatchNormalization(name=bn_name_base + '1')(y)

    x = keras.layers.add([x, y])
    x = keras.layers.Activation('relu')(x)

    return x


# 定义resnet_50网络模型
def resNet_50(input_shape=[224, 224, 3], num_classes=1000):
    img_input = keras.layers.Input(shape=input_shape)
    x = keras.layers.ZeroPadding2D((3, 3))(img_input)

    # 进行卷积和池化
    x = keras.layers.Conv2D(64, (7, 7),strides=(2, 2), name='conv1')(x)
    x = keras.layers.BatchNormalization(name='bn_conv1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)

    # 进入全连接层
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(num_classes, activation='softmax', name='fc1000')(x)

    model = keras.models.Model(img_input, x, name='resnet50')

    model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')

    return model

if __name__ == '__main__':
    model = resNet_50()
    model.summary()
    img_path = 'elephant.jpg'
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = keras.preprocessing.image.array_to_img(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.imagenet_utils.preprocess_input(x)

    print('input img shape: ', x.shape)
    preds = model.predict(x)
    print('Predicted: ', keras.applications.imagenet_utils.decode_predictions(preds))





