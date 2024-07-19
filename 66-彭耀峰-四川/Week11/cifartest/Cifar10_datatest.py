import tensorflow as tf
import os

num_examples_per_epoch_for_train = 5000
num_examples_per_epoch_for_eval = 1000

class CIFAR10Record(object):
    pass

def read_cifar10(file_queue):
    result = CIFAR10Record()

    label_bytes = 1     #1表示Cifar-10,2表示Cifar-100数据集
    result.height = 32
    result.width = 32
    result.depth = 3    #彩色图片RGB通道数为3

    image_bytes = result.height * result.width * result.depth   #图片样本元素总数
    record_bytes = label_bytes + image_bytes    #样本包含图片和标签
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)   #创建一个文件读取类
    result.key,value = reader.read(file_queue)
    record_bytes = tf.decode_raw(value,tf.uint8)   #将文件内容从字符串解析为图像对应的像素数组
    result.label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)   #提取标签
    depth_major = tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes + image_bytes]),
                             [result.depth,result.height,result.width])
    result.uint8image = tf.transpose(depth_major,[1,2,0])   #将图片转换顺序，hwc(即高度、宽度和深度)
    return result



def inputs(data_dir,batch_size,distorted):
    filenames = [os.path.join(data_dir,'data_batch_%d.bin'%i) for i in range(1,6)]
    file_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(file_queue)
    reshaped_image = tf.cast(read_input.uint8image,tf.float32)
    num_examples_per_epoch = num_examples_per_epoch_for_train

    if distorted != None:
        #进行数据增强
        cropped_image = tf.random_crop(reshaped_image,[24,24,3])             #剪切图片
        flipped_image = tf.image.random_flip_left_right(cropped_image)       #左右翻转
        adjusted_brightness = tf.image.random_brightness(flipped_image,max_delta=0.8)   #随机调整亮度
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness,lower=0.2,upper=1.8)   #随机调整对比度
        float_image = tf.image.per_image_standardization(adjusted_contrast)     #将图片标准化
        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_per_epoch_for_eval * 0.4)

        #随机喊声一个批次的image和label
        images_train,labels_train = tf.train.shuffle_batch([float_image,read_input.label],batch_size=batch_size,num_threads=16,
                                capacity=min_queue_examples + 3 * batch_size,
                                min_after_dequeue=min_queue_examples,)

        return images_train,tf.reshape(labels_train,[batch_size])

    else:
        resized_image = tf.image.resize_with_crop_or_pad(reshaped_image,24,24)      #剪切图片
        float_image = tf.image.per_image_standardization(resized_image)     #标准化图像
        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_per_epoch * 0.4)

        images_test,labels_test = tf.train.batch([float_image,read_input.label],batch_size=batch_size,num_threads=16,
                       capacity=min_queue_examples + 3*batch_size)
        return images_test,tf.reshape(labels_test,[batch_size])













