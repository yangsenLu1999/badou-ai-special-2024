import numpy as np
from keras import layers, models

from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

def identity_block(input, kernel_size, filters, stage, block):
    filter1, filter2, filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filter1, (1, 1), name=conv_name_base + '2a')(input)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input])
    x = layers.Activation('relu')(x)
    return x

def conv_block(input, kernel_size, filters, stage, block, strides=(2, 2)):
    filter1, filter2, filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base+'2a')(input)
    x = layers.BatchNormalization(name=bn_name_base+'2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filter2, kernel_size, padding='same', name=conv_name_base+'2b')(x)
    x = layers.BatchNormalization(name=bn_name_base+'2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filter3, (1, 1), name=conv_name_base+'2c')(x)
    x = layers.BatchNormalization(name=bn_name_base+'2c')(x)

    shortcut = layers.Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base+'1')(input)
    shortcut = layers.BatchNormalization(name=bn_name_base+'1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def ResNet50(input_shape=(224, 224, 3), classes=1000):
    img_input = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D((3, 3))(img_input)

    x = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

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

    x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(classes, activation='softmax', name='fc1000')(x)

    # Create model
    model = models.Model(img_input, x, name='resnet50')

    model.load_weights('205-于江龙/week12/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

    return model

if __name__ == '__main__':
    model = ResNet50()
    model.summary()
    img_path = '205-于江龙/week12/resnet50/elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print("Input image shape: ", x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))



