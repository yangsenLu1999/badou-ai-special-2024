import tensorflow as tf
import numpy as np
import os

class yolo:
    def __init__(self,norm_epsilon,norm_decay,anchor_path,class_path,pre_train):
        self.norm_decay=norm_decay   #在预测时计算moving average时的衰减率
        self.norm_epsilon=norm_epsilon   #方差加上极小的数，防止除以0的情况
        self.anchor_path=anchor_path  # yolo anchor 文件路径
        self.class_path=class_path  #数据集类别对应文件
        self.pre_train=pre_train   #是否使用预训练darknet53模型

#获取种类
    def get_class(self):
        class_path=os.path.expanduser(self.class_path)
        #os.path.expanduser---用于将路径字符串中的波浪线（~）扩展为用户的主目录。它的作用是提供跨平台的路径展开功能，使得路径可以在不同的操作系统上正确地解析
        with open(class_path) as f:
            class_names=f.readlines()
        class_names=[c.strip() for c in class_names]  #strip()----将从原始字符串的开头和结尾删除给定的字符。默认情况下，函数strip()将删除字符串开头和结尾的空格
        return  class_names
#获取先验框
    def get_anchors(self):
        anchor_path=os.path.expanduser(self.anchor_path)
        with open(anchor_path) as f:
            anchors=f.readlines()
        anchors=[float(_) for _ in anchors.split(',') ]
        return np.array(anchors).reshape(-1,2)

#bn
    def bn_layer(self,inputs,name=None,training=True,norm_decay = 0.99, norm_epsilon = 1e-3):
        bn_layer=tf.layers.batch_normalization(inputs=inputs,momentum=norm_decay,epsilon=norm_epsilon
                                               ,center=True,scale=True,training=training,name=name)

        return tf.nn.leaky_relu(bn_layer,alpha=0.1)

#conv
    def conv2d_layer(self,inputs,filters,ks,name,use_bias=False,stride=(1,1)):
        conv=tf.layers.conv2d(inputs,filters,stride,kernel_initializer=tf.glorot_uniform_initializer(),
                              padding=('same' if stride==(1,1) else 'valid'),kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),
                          use_bias=use_bias )
        return conv

#residual block 残差卷积块---先进行一次3*3卷积，保存该layer，再进行一次1*1和3*3的卷积，吧结果加上之前的layer
    def Residual_block(self,inputs,filters_num,blocks_num,conv_index,training=True,norm_decay = 0.99, norm_epsilon = 1e-3):
           inputs=tf.pad(inputs,paddings=[[0,0],[1,0],[0,1],[0,0]],mode='CONSTANT')  #tf.pad ---填充函数
           x=self.conv2d_layer(inputs,filters_num,ks=3,stride=(2,2),name='conv2d'+str(conv_index))
           x=self.bn_layer(x,name='bn'+str(conv_index),training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
           conv_index+=1
           for _ in range(blocks_num):
               shortcut=x
               x=self.conv2d_layer(x,filters_num//2,ks=1,stride=(1,1),name='conv2d'+str(conv_index))
               x=self.bn_layer(x,name='bn'+str(conv_index),training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
               conv_index+=1
               x=self.conv2d_layer(x,filters_num,ks=3,stride=(1,1),name='conv2d'+str(conv_index))
               x=self.bn_layer(x,name='bn'+str(conv_index),training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
               conv_index+=1
               x+=shortcut

           return x,conv_index

#构建darknet53模型结构
#Parameters
#----------
#    inputs: 模型输入变量
 #   conv_index: 卷积层数序号，方便根据名字加载预训练权重
 #   weights_dict: 预训练权重
 #   training: 是否为训练
  #  norm_decay: 在预测时计算moving average时的衰减率
  #  norm_epsilon: 方差加上极小的数，防止除以0的情况
#Returns
#-------
   # conv: 经过52层卷积计算之后的结果, 输入图片为416x416x3，则此时输出的结果shape为13x13x1024
   # route1: 返回第26层卷积计算结果52x52x256, 供后续使用
   # route2: 返回第43层卷积计算结果26x26x512, 供后续使用
   # conv_index: 卷积层计数，方便在加载预训练模型时使用

    def darknet53(self,inputs,conv_index,training=True,norm_decay = 0.99, norm_epsilon = 1e-3):
        with tf.variable_scope('darknet53'):  #输入图片大小（416,416,3）
            conv=self.conv2d_layer(inputs,filters=32,ks=3,stride=(1,1),name='con2d'+str(conv_index))  #416,416,32
            conv=self.bn_layer(conv,name='bn'+str(conv_index),training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            conv_index+=1
            conv,conv_index=self.Residual_block(conv,conv_index=conv_index,filters_num=64,blocks_num=1,
                                                training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)  #208,208,64
            conv,conv_index=self.Residual_block(conv,conv_index=conv_index,filters_num=128,blocks_num=2,
                                                training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)  #104,104,128
            conv,conv_index=self.Residual_block(conv,conv_index=conv_index,filters_num=256,blocks_num=8,
                                                training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)  #52,52,256
            route1=conv
            conv,conv_index=self.Residual_block(conv,conv_index=conv_index,filters_num=512,blocks_num=8,
                                                training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)  #26,26,,512
            route2=conv
            conv,conv_index=self.Residual_block(conv,conv_index=conv_index,filters_num=1024,blocks_num=4,
                                                training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)  #13,13,1024
            return route1,route2,conv,conv_index

#yolo3在Darknet53提取的特征层基础上，又加了针对3种不同比例的feature map的block，这样来提高对小物体的检测率
    def yolo_block(self,inputs,filters_num,out_filters,conv_index,training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        conv = self.conv2d_layer(inputs,filters = filters_num, ks = 1, stride = (1,1), name = "conv2d_" + str(conv_index))
        conv = self.bn_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters = filters_num * 2, ks = 3, stride =(1,1), name = "conv2d_" + str(conv_index))
        conv = self.bn_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters = filters_num, ks = 1, stride = (1,1), name = "conv2d_" + str(conv_index))
        conv = self.bn_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters = filters_num * 2, ks = 3, stride = (1,1), name = "conv2d_" + str(conv_index))
        conv = self.bn_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters = filters_num, ks = 1, stride =(1,1), name = "conv2d_" + str(conv_index))
        conv = self.bn_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        route = conv
        conv = self.conv2d_layer(conv, filters = filters_num * 2, ks = 3, stride =(1,1), name = "conv2d_" + str(conv_index))
        conv = self.bn_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters = out_filters, ks = 1, stride = (1,1), name = "conv2d_" + str(conv_index), use_bias = True)
        conv_index += 1
        return route, conv, conv_index

#构建yolo模型结构
    def yolo_inference(self,inputs,num_anchors,num_classes,training=True):
        """
        Introduction
        ------------
            构建yolo模型结构
        Parameters
        ----------
            inputs: 模型的输入变量
            num_anchors: 每个grid cell负责检测的anchor数量
            num_classes: 类别数量
            training: 是否为训练模式
        """
        conv_index=1
        # route1 = 52,52,256、route2 = 26,26,512、route3 = 13,13,1024
        conv2d_26,conv2d_43,conv,conv_index=self.darknet53(inputs,conv_index,training=training,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
        with tf.variable_scope('yolo'):  #获取第一个特征层
            conv2d_57,conv2d_59,conv_index=self.yolo_block(conv,512,num_anchors*(num_classes+5),conv_index=conv_index,
                                                           training=training,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
            ## conv2d_57 = 13,13,512，conv2d_59 = 13,13,255(3x(80+5))
            conv2d_60=self.conv2d_layer(conv2d_57,filters=256,ks=1,stride=(1,1),name='conv2d'+str(conv_index)) #获取第二个特征层
            conv2d_60=self.bn_layer(conv2d_60,name='bn'+str(conv_index),
                                    training=training,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
            conv_index+=1
            # unSample_0 = 26,26,256
            unSample_0=tf.image.resize_nearest_neighbor(conv2d_60,[2*tf.shape(conv2d_60)[1],2*tf.shape(conv2d_60)[1]],name='unSample_0')
            rount0=tf.concat([unSample_0,conv2d_43],axis=-1,name='rount0')  #route0 = 26,26,768
            conv2d_65,conv2d_67,conv_index=self.yolo_block(rount0,256,num_anchors*(num_classes+5),conv_index=conv_index,
                                                           training=training,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)

            conv2d_68=self.conv2d_layer(conv2d_65,filters=128,ks=1,stride=(1,1),name='conv2d'+str(conv_index))  #获取第三个特征层
            conv2d_68=self.bn_layer(conv2d_68,name='bn'+str(conv_index),training=training,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
            conv_index+=1
            unSample_1=tf.image.resize_nearest_neighbor(conv2d_68,[2*tf.shape(conv2d_60)[1],2*tf.shape(conv2d_60)[1]],name='unSample_1')
            rount1=tf.concat([unSample_1,conv2d_68],axis=-1,name='rount1')
            _n,conv2d_75,_m=self.yolo_block(rount1,128,num_anchors*(num_classes+5),conv_index=conv_index,
                                                           training=training,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
        return [conv2d_59,conv2d_67,conv2d_75]




















