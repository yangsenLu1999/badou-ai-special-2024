import numpy as np
import tensorflow as tf
from PIL import Image

# 加载预训练好的darknet53权重文件
def load_weights(var_list,weights_file):
    with open(weights_file,'r') as fp:
        weights=np.fromfile(fp,dtype=np.float32)   #np.fromfile 是 NumPy 库中的一个函数，它用于直接从二进制文件中读取数据并将其作为 NumPy 数组加载

    ptr=0
    i=0
    assign_ops=[]
    while i<len(var_list)-1:
        var1=var_list[0]
        var2=var_list[1]

        if 'con2d' in var1.name.split('/')[-2]:
            if 'batch_normalization' in var1.name.split('/')[-2]:
                gamma,beta,mean,var=var_list[i+1:i+5]
                batch_norm_var=[beta,gamma,mean,var]
                for var in  batch_norm_var:
                    shape=var.shape.as_list()
                    num_params=np.prod(shape)      #np.prod()函数用来计算所有元素的乘积
                    var_weights=weights[ptr:ptr+num_params].reshape(shape)
                    ptr+=num_params
                    assign_ops.append(tf.assign(var,var_weights,validate_shape=True))            # tf.assign在tensorflow起到赋值的作用,对g进行assign后g的值就变成新的值,同时返回的也是新的值

                i+=4
            elif 'conv2d' in var2.name.split('/')[-2]:
                bias=var2
                bias_shape=bias.shape.as_list()
                bias_params=np.prod(bias_shape)
                bias_weigths=weights[ptr:ptr+bias_params].reshape(bias_shape)
                ptr+=bias_params
                assign_ops.append(tf.assign(bias,bias_weigths,validate_shape=True))
                i+=1

            shape=var1.shape.as_list()
            num_params=np.prod(shape)
            var_weights=weights[ptr:ptr+num_params].reshape(shape[3],shape[2],shape[0],shape[1])
            ptr+=num_params
            assign_ops.append(tf.assign(var1,var_weights,validate_shape=True))
            i+=1

    return assign_ops


#对预测输入图像进行缩放，按照长宽比进行缩放，不足的地方进行填充
def letterbox_image(image,size):
    image_w,image_h=image.size
    w,h=size
    new_w=int(image_w*min(w*1.0/image_w,h*1./image_h))
    new_h=int(image_h*min(w*1.0/image_w,h*1./image_h))
    resize_image=image.resize((new_w,new_h), Image.BICUBIC)   #三线性插值

    boxed_image=Image.new('RGB',size,(128,128,128))
    boxed_image.paste(resize_image,((w-new_h)//2,(h-new_h)//2))   #python中PIL库中的paste函数的作用为将一张图片覆盖到另一张图片的指定位置去

    return boxed_image

#画框函数
def draw_box(image,bbox):
    xmin,ymin,xmax,ymax,label=tf.split(value=bbox,num_or_size_splits=5,axis=2)
    h=tf.cast(tf.shape(image)[1])
    w=tf.cast(tf.shape(image)[2])
    new_box=tf.concat([tf.cast(xmin,tf.float32)/h,tf.cast(ymin,tf.float32)/w,tf.cast(xmax,tf.float32)/h,tf.cast(ymax,tf.float32)/w],2)
    new_image=tf.image.draw_bounding_boxes(image,new_box)
    tf.summary.image('input',new_image)

#通过召回率和精确度算AP（平均精度）
def voc_ap(rec,prec):
     rec.insert(0.0)
     rec.append(1.0)
     mrec=rec[:]
     prec.insert(0.0)
     prec.append(0.0)
     mpre=prec[:]
     for i in range(len(mpre)-2,-1,-1):
         mpre[i]=max(mpre[i],mpre[i+1])
     i_list=[]
     for i in range(1,len(mrec)):
         if mrec[i]!=mrec[i-1]:
             i_list.append(i)
     ap=0.0
     for i in i_list:
         ap+=((mrec[i]-mrec[i-1]*mpre[i]))
     return ap,mrec,mpre















