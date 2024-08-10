
from keras import layers
from keras.layers import Dense,Conv2D,BatchNormalization,Activation,MaxPooling2D,Input,ZeroPadding2D,AveragePooling2D,Flatten,Softmax
import numpy as np
from keras.models import Model

# 剩余部分
from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import  preprocess_input


def identity_block(input_tensor,kernel_size,filters,stage,block):

    # 这个块直接并行连接 不用对图片大小进行缩放 自然不用设置步长等其他东西
    filters1, filters2, filters3 = filters

    # 直接写 block 名字 每个bolock名字 方便后续打印理解层数
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),  name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)


    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),  name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)



    x = layers.add([x,input_tensor])
    x = Activation('relu')(x)


    return x




def conv_block(input_tensor,kernel_size,filters,stage,block,strides=(2,2)):

    filters1,filters2,filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'


    # conv bolock 结构  输入的x input 需要经过一个卷积进行
    x = Conv2D(filters1,(1,1),strides=strides,name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size=kernel_size, name=conv_name_base + '2b',padding='same')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3,(1,1),name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base+'2c')(x)
    # 这是第二个卷积块 ，要把它加上经过卷积后的 input block 之后actviaation

    # shortcut 并联接入  注意这里的strides 需要设置 保证shape一样
    shortcut = Conv2D(filters3,(1,1),strides=strides,name=conv_name_base+'1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)


    # 拼接操作
    x = layers.add([x,shortcut])
    x = Activation('relu')(x)

    return  x



def ResNet50(input_size=[24,24,3],classes=1000):

    img_input = Input(shape=[224,224,3])
    x = ZeroPadding2D((3,3))(img_input)

    x = Conv2D(64,(7,7),strides=(2,2),name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3),strides=(2,2))(x)


    x = conv_block(x,3,[64,64,256],stage=2,block='a',strides=(1,1))
    x = identity_block(x,3,[64,64,256],stage=2,block='b')
    x = identity_block(x,3,[64,64,256],stage=2,block='c')

    # 下一大层
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 第二层
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')


    #第三层
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')


   #  最后池化全连接
    x = AveragePooling2D((7,7),name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(classes,activation='softmax',name='fc1000')(x)


    # 利用keras model
    model = Model(img_input,x,name='resnet50')

    # 拿出训练好的模型
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model

if __name__ == '__main__':

    model = ResNet50()

    # 显示模型结构 等其他结果
    model.summary()

    img_dir = 'elephant.jpg'

    img = image.load_img(img_dir,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)

    print('orgin_input_shape: ',x.shape)
    preds = model.predict(x)
    print('Predicted: ',decode_predictions(preds))















