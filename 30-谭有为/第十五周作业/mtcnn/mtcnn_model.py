from keras.layers import Input,Conv2D,MaxPool2D,Permute,Flatten,Dense
from keras.layers.advanced_activations import PReLU
from keras.models import Model
import utils as u
import cv2
import numpy as np

#p-net---粗略获取人脸框输出bbox位置以及是否有人脸
def p_net(weight_path):
    input=Input(shape=(None,None,3))   #12,12,3

    x=Conv2D(10,kernel_size=(3,3),strides=1,padding='valid',name='conv1')(input)  #9,9,10
    #shared_axes: 激活函数共享可学习参数的轴。 例如，如果输入特征图来自输出形状为(batch, height, width, channels)的2D
    #卷积层，而且你希望跨空间共享参数，以便每个滤波器只有一组参数， 可设置shared_axes = [1, 2]。
    x=PReLU(shared_axes=[1,2],name='prelu1')(x)
    x=MaxPool2D(pool_size=2)(x)    #5,5,10

    x=Conv2D(16,kernel_size=(3,3),strides=1,padding='valid',name='conv2')(x)  #3,3,16
    x=PReLU(shared_axes=[1,2],name='prelu2')(x)

    x=Conv2D(32,kernel_size=(3,3),strides=1,padding='valid',name='conv3')(x)  #1,1,32
    x=PReLU(shared_axes=[1,2],name='prelu3')(x)

    classification=Conv2D(2,kernel_size=1,activation='softmax',name='conv4-1')(x)
    #线性，无激活函数，两个结果--是人脸概率&不是人脸概率
    bbox_regression=Conv2D(4,kernel_size=1,name='conv4-2')(x)
    #线性，无激活函数，4个结果--左上角坐标、右下角坐标、人脸框的宽、人脸框的高

    model=Model([input]
                ,[classification,bbox_regression])
    model.load_weights(weight_path,by_name=True)  #by_name=True，只加载权重中name匹配的层

    return model


#r-net---精细调整人脸框位置以及是否有人脸
def r_net(weight_path):
    input=Input(shape=([24,24,3]))   #24,24,3

    x=Conv2D(28,kernel_size=(3,3),strides=1,padding='valid',name='conv1')(input)  #22,22,28
    x=PReLU(shared_axes=[1,2],name='prelu1')(x)
    x=MaxPool2D(pool_size=3,strides=2,padding='same')(x)    #11,11,28

    x=Conv2D(48,kernel_size=(3,3),strides=1,padding='valid',name='conv2')(x)  #9,9,48
    x=PReLU(shared_axes=[1,2],name='prelu2')(x)
    x=MaxPool2D(pool_size=3,strides=2)(x)    #4,4,48

    x=Conv2D(64,kernel_size=(2,2),strides=1,padding='valid',name='conv3')(x)  #3,3,64
    x=PReLU(shared_axes=[1,2],name='prelu3')(x)

    x=Permute(dims=(3,2,1))(x)   #Permute---重新排列输入张量的维度顺序 ,64,3,3
    x=Flatten()(x)   #576

    x=Dense(128,name='conv4')(x)  #128
    x=PReLU(name='prelu4')(x)

    classification=Dense(2,activation='softmax',name='conv5-1')(x)  #2个结果--是人脸概率&不是人脸概率
    bbox_regression=Dense(4,name='conv5-2')(x)  #4个结果--左上角坐标、右下角坐标、人脸框的宽、人脸框的高


    model=Model([input],[classification,bbox_regression])

    model.load_weights(weight_path,by_name=True)

    return model


#o-net---进一步精细调整人脸框位置以及是否有人脸
def o_net(weight_path):
    input=Input(shape=(48,48,3))   #48,48,3

    x=Conv2D(32,kernel_size=(3,3),strides=1,padding='valid',name='conv1')(input)  #46,46,32
    x=PReLU(shared_axes=[1,2],name='prelu1')(x)
    x=MaxPool2D(pool_size=3,strides=2,padding='same')(x)    #23,23,32

    x=Conv2D(64,kernel_size=(3,3),strides=1,padding='valid',name='conv2')(x)  #21,21,64
    x=PReLU(shared_axes=[1,2],name='prelu2')(x)
    x=MaxPool2D(pool_size=3,strides=2)(x)    #10,10,64

    x=Conv2D(64,kernel_size=(3,3),strides=1,padding='valid',name='conv3')(x)  #8,8,64
    x=PReLU(shared_axes=[1,2],name='prelu3')(x)
    x=MaxPool2D(pool_size=2,strides=2,padding='same')(x)    #4,4,64

    x=Conv2D(128,kernel_size=(2,2),strides=1,padding='valid',name='conv4')(x)  #3,3,128
    x=PReLU(shared_axes=[1,2],name='prelu4')(x)

    x=Permute(dims=(3,2,1)) (x)   #128,3,3
    x=Flatten()(x)   #1152

    x=Dense(256,name='conv5')(x)  #256
    x=PReLU(name='prelu5')(x)

    classification=Dense(2,activation='softmax',name='conv6-1')(x)  #2个结果--是人脸概率&不是人脸概率
    bbox_regression=Dense(4,name='conv6-2')(x)  #4个结果--左上角坐标、右下角坐标、人脸框的宽、人脸框的高
    landmark_regression=Dense(10,name='conv6-3')(x)  #10个结果--五官坐标

    model=Model([input],[classification,bbox_regression,landmark_regression])
    model.load_weights(weight_path,by_name=True)
    return model


class MTCNN:
    def __init__(self):
        self.pnet=p_net('weights/pnet.h5')
        self.rnet=r_net('weights/rnet.h5')
        self.onet=o_net('weights/onet.h5')


    def detect_face(self,img,threshold):
        copy_img=(img.copy()-127.5)/127.5  #归一化,[0,255]映射到[-1,1]，加快收敛速度
        origin_h,origin_w,_=copy_img.shape

        scales=u.Scale(img)  #得到图像金字塔
        out_boxes=[]

        #pnet过程
        for scale in scales:
            hs=int(origin_h*scale)
            ws=int(origin_w*scale)
            scale_img=cv2.resize(copy_img,(ws,hs))
            inputs=scale_img.reshape(1,*scale_img.shape)
            output=self.pnet.predict(inputs)  #将图像金字塔传入pnet
            out_boxes.append(output)   #将图片传入pnet得到预测结果
            #print("pnet:",out_boxes)

        image_num=len(scales)
        rectangles=[]
        for i in range(image_num):
           cls_prob=out_boxes[i][0][0][:,:,1]  #取有人脸的概率
           rio=out_boxes[i][1][0]   #取人脸框坐标偏移量

           out_h,out_w=cls_prob.shape
           out_side=max(out_h,out_w)

           #解码过程
           rectangle=u.detect_face_pnet(cls_prob,rio,out_side,1/scales[i],origin_w,origin_h,threshold[0])  #得到pnet预测框
           rectangles.extend(rectangle)  #将pnet预测框加入rectangles  extend()方法用于在列表末尾一次性追加另一个序列中的多个值。
        #NMS
        rectangles=u.NMS(rectangles,0.7)
        print("pnet-nms:",rectangles)


        if len(rectangles)==0:
            return rectangles   #   如果没有检测到人脸，则返回空列表

        #rnet过程
        pridict_rnet_batch=[]
        for rectangle in rectangles:
            crop_img=copy_img[int(rectangle[1]):int(rectangle[3]),int(rectangle[0]):int(rectangle[2])]  #裁剪出人脸区域
            scale_img=cv2.resize(crop_img,(24,24))  #缩放为24*24
            pridict_rnet_batch.append(scale_img)  # 将裁剪后的人脸区域加入列表,即rnet的输入

        pridict_rnet_batch=np.array(pridict_rnet_batch)
        out=self.rnet.predict(pridict_rnet_batch)  #将rnet的输入传入rnet,得到rnet输出

        cls_prob=out[0]  #   取有人脸的概率
        cls_prob=np.array(cls_prob)
        rio_prob=out[1]   #   取人脸框坐标偏移量
        rio=np.array(rio_prob)
        rectangles=u.detect_face_rnet(cls_prob,rio_prob,rectangles,origin_w,origin_h,threshold[1])  #   得到rnet预测框

        if len(rectangles)==0:
            return rectangles   #   如果没有检测到人脸，则返回空列表

        #onet过程
        pridict_onet_batch=[]
        for rectangle in rectangles:
            crop_img=copy_img[int(rectangle[1]):int(rectangle[3]),int(rectangle[0]):int(rectangle[2])]  #裁剪出人脸区域
            scale_img=cv2.resize(crop_img,(48,48))  #缩放为48*48
            pridict_onet_batch.append(scale_img)  # 将裁剪后的人脸区域加入列表,即onet的输入

        pridict_onet_batch=np.array(pridict_onet_batch)
        out_put=self.onet.predict(pridict_onet_batch)  #将onet的输入传入onet,得到onet输出
        #print("onet-out_put:",out_put)

        cls_prob=out_put[0]  #   得到有人脸的概率
        rio_prob=out_put[1]   #   得到人脸框坐标偏移量
        pts_prob=out_put[2]   #   得到五官坐标
       # print("cls_prob:",cls_prob)
        rectangles=u.detect_face_onet(cls_prob,rio_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])  #   得到onet预测框

        return rectangles   #   返回检测到的人脸框































