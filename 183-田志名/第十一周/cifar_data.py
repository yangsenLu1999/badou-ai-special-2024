import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
#参数设计
classes=10
train_data_num=50000
test_data_num=10000

#定义一个空类，用于存放数据信息及返回
class CIFAR10Record(object):
    pass

#读取数据，将数据的各个信息写到类CIFAR10Record中，最后将这个类返回
def read_cifar10(file_queue):
    #需要从二进制文件中读取信息，其中第一个比特存放标签，后面存放图片的信息
    result = CIFAR10Record()                      #实例化类CIFAR10Record，这个类用于存放各种信息
    label_byte=1                                  #10分类任务，一个byte便可以实现标签
    result.h=32
    result.w=32
    result.c=3                                    #图片高度，长度以及通道数

    imgae_btyes=result.h*result.w*result.c        #一张图片的比特数目
    smaple_bytes=imgae_btyes+label_byte           #一个样本的bytes，图片加标签

    reader=tf.FixedLengthRecordReader(record_bytes=smaple_bytes)     #构建一个文件读取的类，本质是一个迭代器，每次读取smaple_bytes
    result.key, value = reader.read(file_queue)  # 使用该类的read()函数从文件队列里面读取文件
    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label=tf.strided_slice(record_bytes,[0],[label_byte])   #tf.strided_slice对张量进行切片操作
    result.label=tf.cast(result.label,tf.int32)


    result.data=tf.strided_slice(record_bytes,[label_byte],[label_byte+imgae_btyes])
    result.data=tf.reshape(result.data,[result.c,result.h,result.w])        #先构建成3*28*28的，然后再转置，变成3*28*28，此时不能再用reshape
    result.data=tf.transpose(result.data,[1,2,0])                           #通过转置,换成28*28*3的形式（模型要求）
    '''
    tf.strided_slice(record_bytes,[0],[label_byte])   tf.strided_slice对张量进行切片操作
    pytorch:torch.narrow用于在指定的维度上对张量进行切片，可以选择起始索引、结束索引
    result = torch.narrow(x, 0, 1, 2).在第0维上，从索引1开始，到索引3（不包括）结束
    
    tf.cast
    pytroch:torch.tensor(x,dtype=torch.int)

    tf.transpose
    pytroch:x.permute用于重新排列张量的维度顺序。或者torch.transpose(x, 0, 1)
    '''

    return result

def inputs(data_dir,batch_size,flag):
    file_queue=[os.path.join(data_dir,"data_batch_%d.bin"%i) for i in range(1,6)]   #存放5个文件地址
    file_queue = tf.train.string_input_producer(file_queue)  # 根据已经有的文件地址创建一个文件队列,与tf.FixedLengthRecordReader(record_bytes=smaple_bytes)搭配使用
    #读取信息
    print(file_queue)
    read_input = read_cifar10(file_queue)                    #read_input是一个类
    reshaped_image = tf.cast(read_input.data, tf.float32)    #类型转换

    #对于训练集，图片增强
    if flag:
        cropped_image=tf.random_crop(reshaped_image,[24,24,3])          #首先将预处理好的图片进行剪切，使用tf.random_crop()函数
        flipped_image=tf.image.random_flip_left_right(cropped_image)    #将剪切好的图片进行左右翻转，使用tf.image.random_flip_left_right()函数
        adjusted_brightness=tf.image.random_brightness(flipped_image,max_delta=0.8)   #将左右翻转好的图片进行随机亮度调整，使用tf.image.random_brightness()函数
        adjusted_contrast=tf.image.random_contrast(adjusted_brightness,lower=0.2,upper=1.8)    #将亮度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数
        float_image=tf.image.per_image_standardization(adjusted_contrast)          #进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差

        '''
        tf.random_crop()是TensorFlow中的一个函数，用于从输入张量中随机裁剪出一个指定大小的区域
        pytorch中和他一样的是torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
        
        tf.image.random_flip_left_right是TensorFlow中的一个函数，用于随机地左右翻转输入的图像
        pytorch中:torchvision.transforms.functional.rotate(image, angle, expand=False, center=None, fill=0, interpolation=None)
        
        tf.image.random_brightness是TensorFlow中的一个函数，用于随机调整图像的亮度。max_delta=0.8,正值增加亮度，负值降低
        pytorch:torchvision.transforms.ColorJitter(brightness=0.5)，brightness=0.5表示亮度将在原始值的基础上随机增加或减少最多50%。
        
        tf.image.random_contrast函数用于随机调整图像的对比度。它接受一个图像张量作为输入，并返回一个具有随机对比度调整的新图像张量。
        pytorch:torchvision.transforms.ColorJitter（contrast=0.5），contrast=0.5表示对比度将在原始值的基础上随机增加或减少最多50%。
        '''

        float_image.set_shape([24,24,3])                      #设置图片数据及标签的形状
        print(float_image)
        read_input.label.set_shape([1])
        min_queue_examples = int(train_data_num * 0.4)

        #下面这个代码会调用多次read_cifar10,每个样本都会执行一次，其中的reader作为迭代器，也是计算图的一部分，不会释放，每次取出一个样本数据。
        #tf.train.shuffle_batch会收集capacity个样本，然后选出来batch_size个。
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples,
                                                            )
        '''
        tf.train.shuffle_batch是TensorFlow中的一个函数，用于从输入数据集中随机抽取一批样本，并对这些样本进行打乱。
        tensors: 一个包含张量的列表，每个张量代表一个输入数据集的一个元素。
        batch_size: 每个批次的样本数量。
        capacity: 队列的最大容量。
        min_after_dequeue: 出队后队列中剩余元素的最小数量，用于确保随机性。
        在创建随机打乱队列时，可以通过设置min_after_dequeue参数来指定队列中元素的最小数量。这个参数的作用是在每次出队操作后，
        确保队列中至少有min_after_dequeue个元素。这样可以保证在随机打乱过程中，队列中的元素不会因为频繁的出队操作而变得稀疏。
        '''
        return images_train,tf.reshape(labels_train,[batch_size])
    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24,24)  # 在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切
        float_image = tf.image.per_image_standardization(resized_image)  # 剪切完成以后，直接进行图片标准化操作
        float_image.set_shape([24, 24, 3])        #和tf.reshape一致,24*24*3
        read_input.label.set_shape([1])

        min_queue_examples = int(test_data_num * 0.4)
        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size, num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)
        return images_test, tf.reshape(labels_test, [batch_size])


if __name__=="__main__":
    images_train,labels_train=inputs("Cifar_data/cifar-10-batches-bin",100,1)
    with tf.Session() as sess:
        tf.train.start_queue_runners()     #不开新线程的话，会卡死，不会去执行下面的代码，因为上面数据处理开了16个线程
        image_batch, label_batch = sess.run([images_train, labels_train])
        print(image_batch.shape)    #100*24*24*3

