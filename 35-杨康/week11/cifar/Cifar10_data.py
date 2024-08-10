import tensorflow as tf
import os
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000
class CIFAR10Record(object):
    pass

def read_cifar10(file_queue):
    result = CIFAR10Record()
    label_bytes = 1   #cifar10只要1位0-9，100需要2位0-99
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height* result.width*result.depth
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes= record_bytes)  #使用tf.FixedLengthRecordReader()创建一个文件读取类。该类的目的就是读取文件
    result.key,value = reader.read(file_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)  # 读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组
    # 因为该数组第一个元素是标签，所以我们使用strided_slice()函数将标签提取出来，并且使用tf.cast()函数将这一个标签转换成int32的数值形式
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    # 剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
    # 这一步是将一维数据转换成3维数据
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result

def inputs(data_dir,batch_size,distorted):  #distorted用于是否需要图像增强
    file_name = [os.path.join(data_dir,'data_batch_%d.bin'%i)for i in range(1,6)]
    file_queue = tf.train.string_input_producer(file_name)  #生成队列文件
    read_input = read_cifar10(file_queue)
    reshaped_image = tf.cast(read_input.uint8image,tf.float32) #转换数据格式
    num_examples_per_epoch = num_examples_pre_epoch_for_train
    if distorted != None:           #需要图像增强
        image_crop = tf.random_crop(reshaped_image,[24,24,3])  #随机裁剪图片为[24,24,3]
        image_flip = tf.image.random_flip_left_right(image_crop) #图片左右翻转
        image_brightness = tf.image.random_brightness(image_flip,max_delta=0.8) #图片亮度
        image_contrast = tf.image.random_contrast(image_brightness,lower=0.2,upper=1.8) #图片对比度
        float_image = tf.image.per_image_standardization(image_contrast)    #进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差
        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_eval*0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)
        image_train,image_label = tf.train.shuffle_batch([float_image,read_input.label],
                                                         batch_size=batch_size,
                                                         num_threads=16,
                                                         capacity=min_queue_examples + 3*batch_size,
                                                         min_after_dequeue=min_queue_examples)
                                            # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label
        return image_train, tf.reshape(image_label,[batch_size])
    else:           #不需要图像增强
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,24,24)   #在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切
        float_image = tf.image.per_image_standardization(resized_image)
        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size, num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)
        # 这里使用batch()函数代替tf.train.shuffle_batch()函数
        return images_test, tf.reshape(labels_test, [batch_size])