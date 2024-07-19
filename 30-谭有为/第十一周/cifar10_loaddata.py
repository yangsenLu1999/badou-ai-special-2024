import tensorflow as  tf
import  os


#数据初始化
num_class=10
epoch_num_for_train=50000
epoch_num_for_eva=10000

#定义空类 返回读取的cifar10数据
class Cifar10Record(object):
    pass

#读取cifar10数据
def get_cifar10(file_path):
    res=Cifar10Record()
    label_byte=1    #cifar10对应的标签10个  0-9 1个占位符就够 如果是cifar100 则为2
    res.h,res.w,res.c=32,32,3  #定义输入图片的尺寸
    img_byte=res.h*res.w*res.c  #一张图片的像素总数
    all_byte=img_byte+label_byte #像素总数+标签个数
# tf.FixedLengthRecordReader,读取固定长度字节数信息(针对bin文件使用FixedLengthRecordReader读取比较合适)
    reader=tf.FixedLengthRecordReader(record_bytes=all_byte)  ##使用tf.FixedLengthRecordReader()创建一个文件读取类。该类的目的就是读取文件
    res.key,value=reader.read(file_path)  #根据文件路径读取文件，读取的数据类型为字符串
#tf.decode_raw的意思是将原来编码为字符串类型的变量重新变回来
 #  all_byte=tf.cast(value,tf.uint8)
    all_byte=tf.decode_raw(value,tf.uint8)
#tf.strided_slice--从给定的 input_ 张量中提取一个尺寸 (end-begin)/stride 的片段
    res.label=tf.reshape(tf.cast(tf.strided_slice(all_byte,[0],[label_byte]),tf.int32),[1,1])  #提取标签并数据转换
    res.pixels=tf.cast(tf.strided_slice(all_byte,[label_byte],[img_byte+label_byte]),tf.int32) #提取图片像素点并数据转换
    res.pixels=tf.reshape(res.pixels,[res.c,res.h,res.w])  #reshape 一维转三维
    res.uint8image=tf.transpose(res.pixels,[1,2,0])  #CHW转换成HWC

    return res

#数据预处理
def input_datas(data_path,batch_size,is_distored):
    filename=[os.path.join(data_path,"data_batch_%d.bin"%i)for i in range(1,6)]  # os.path.join用于将多个路径拼接为一个完整路径
#将文件名列表交给tf.train.string_input_producer函数，string_input_producer会生成一个先入先出的队列， 文件阅读器会需要它来读取数据
    file_path=tf.train.string_input_producer(filename)
    read_input=get_cifar10(file_path)
    reshape_imgs=tf.cast(read_input.uint8image,tf.float32)  ##将已经转换好的图片数据再次转换为float32的形式
    epoch_size=epoch_num_for_train


   #图像增强   参考文档：https://blog.csdn.net/akadiao/article/details/78541763
    if is_distored!=None:
       crop_imgs=tf.random_crop(reshape_imgs,[24,24,3])    #随机裁剪函数random_crop
       filp_imgs=tf.image.flip_left_right(crop_imgs)     #tf.image.flip_left_right---将图像左右翻转
       bright_imgs=tf.image.random_brightness(filp_imgs,max_delta=0.8)  #tf.image.random_brightness--随机调整图像亮度
       contrast_imgs=tf.image.random_contrast(bright_imgs,lower=0.2,upper=1.8)  #tf.image.random_contrast--随机调整对比度
       standardization_imgs=tf.image.per_image_standardization(contrast_imgs)  #tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差
       standardization_imgs.set_shape=([24,24,3])    #设置图片数据及标签的形状
       read_input.label.set_shape=([1])
       min_data_size=int(epoch_num_for_eva*0.4)  #设置batch数据量
       print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              %min_data_size)
       imgs_train,labels_train=tf.train.shuffle_batch([standardization_imgs,read_input.label],batch_size=batch_size,
                                                         num_threads=16,
                                                         capacity=min_data_size + 3 * batch_size,
                                                         min_after_dequeue=min_data_size,)
       return imgs_train,tf.reshape(labels_train,[batch_size])
                             #使用tf.train.shuffle_batch()函数随机产生一个batch的image和label


#tf.train.shuffle_batch是将队列中的数据随机打乱后再读取出来  参数含义：batch_size：‌单次批处理的样本数量，‌即每个batch中包含的样本个数。‌
       #num_threads：‌线程数量，‌表示用于数据处理的线程数。‌capacity：‌整个队列的容量，‌表示队列中可以存储的最大样本数。‌
    else:
# 通过tf.image.resize_image_with_crop_or_pad函数调整图像的大小。这个函数的第一个参数为原始图像，后面两个参数是调整后
    # 的目标图像大小。如果原始图像的尺寸大于目标图像，那么这个函数会自动截取原始图像中剧中的部分。如果目标图像大于原始图
    # 像，这个函数会自动在原始图像的四周填充全0背景
        resize_imgs=tf.image.resize_image_with_crop_or_pad(reshape_imgs,24,24)
        standardization_imgs=tf.image.per_image_standardization(resize_imgs)
        standardization_imgs.set_shape([24,24,3])
        read_input.label.set_shape=([1])
        min_data_size=int(epoch_size*0.3)
        imgs_train,labels_train=tf.train.shuffle_batch([standardization_imgs,read_input.label],batch_size=batch_size,
                                                          num_threads=10,capacity=(min_data_size+3*batch_size),min_after_dequeue=min_data_size)
        return imgs_train,tf.reshape(labels_train,[batch_size])
















