from keras.layers  import Dense,ZeroPadding2D,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization,Activation,AveragePooling2D,Input
from keras import layers
from keras import Model
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions,preprocess_input


def identity_block(inputs,kernel_size,filters,stage,block):
    filters1,filters2,filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1,(1,1), name=conv_name_base + '2a')(inputs)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x= layers.add([x, inputs])
    x =  Activation('relu')(x)
    return x;


def conv_block(inputs,kernel_size,filters,stage,block,strides=(2, 2)):
    filters1,filters2,filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(inputs)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    y = Conv2D(filters3, (1,1), name=conv_name_base+ '1',strides=strides)(inputs)
    y = BatchNormalization(name=bn_name_base + '1')(y)

    x = layers.add([x,y])
    x = Activation('relu')(x)
    return x

def ResNet50(input_shape=[224,224,3],classes=1000):
    input = Input(shape=input_shape)
    x = ZeroPadding2D((3,3))(input)

    x = Conv2D(64,strides=(2,2),kernel_size=(7,7),name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = conv_block(x,3,[64,64,256],stage=2,block='a',strides=(1,1))
    x = identity_block(x, 3,[64,64,256],stage=2,block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x,3,[128,128,512],stage=3,block='a')
    x = identity_block(x, 3,[128,128,512],stage=3,block='b')
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

    x=AveragePooling2D((7,7),name='avg_pool')(x)
    x = Flatten()(x)

    x =Dense(classes, activation='softmax',name='fc1000')(x)
    model = Model(input,x,name='resnet50')
    model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    return model


if __name__ =='__main__':
    model = ResNet50()
    model.summary()
    #加载图片
    img = image.load_img('bike.jpg', target_size=(224,224))
    #图片 转array
    x = image.img_to_array(img)
    #扩展维度
    x = np.expand_dims(x, axis=0)
    #预处理图片
    x = preprocess_input(x)

    print("图片形状：",x.shape)
    #预测
    preds = model.predict(x)

    print("preds:",decode_predictions(preds))
