import os
import tensorflow as tf
num_class = 10

#设定用于训练和评估的样本总数
num_example_per_epoch_for_train = 50000
num_example_per_epoch_for_eval = 10000

#定义一个空类，用于返回读取的Cifar-10的数据
class CIFAR10Record(object):
    pass

def read_cifar10(file_queue):
    result = CIFAR10Record()

    label_bytes = 1     #label_bytes=1 : Cifar-10; label_bytes=2 : Cifar-100
    result.height = 32
    result.width = 32
    result.depth = 3    #RGB三通道

    image_bytes = result.height * result.width * result.depth  #图片样本总元素数量
    record_bytes = label_bytes + image_bytes #最终样本总数量 = 图片样本总元素数量 + 标签样本总数

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes) #使用tf.FixedLengthRecordReader()创建一个文件读取类。该类的目的就是读取文件
    result.key, value = reader.read(file_queue) #使用该类的read()函数从文件队列里面读取文件

    record_bytes = tf.decode_raw(value, tf.uint8)  #读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组

    # 因为该数组第一个元素是标签，所以我们使用strided_slice()函数将标签提取出来，并且使用tf.cast()函数将这一个标签转换成int32的数值形式
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    #剩下的元素就是图片数据
    #将一维数据depth * height * width，转换为3维数据[depth, height, width]
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])

    #将分割好的图片数据使用tf.transpose()函数转换数据排布方式为CHW
    result.uint8image = tf.transpose(depth_major, [1,2,0])

    return result

 #这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
def inputs(data_dir, batch_size, distorted):
    filename = [os.path.join(data_dir, "data_batch_%d.bin"%i) for i in range(1,6)] #拼接地址
    file_queue = tf.train.string_input_producer(filename) #根据已经有的文件地址创建一个文件队列
    read_input = read_cifar10(file_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32) #将已经转换好的图片数据再次转换为float32的形式
    num_example_per_epoch = num_example_per_epoch_for_train

    # 如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理
    if distorted != None:
        # 首先将预处理好的图片进行剪切，使用tf.random_crop()函数
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
        # 将剪切好的图片进行左右翻转，使用tf.image.random_flip_left_right()函数
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        # 将左右翻转好的图片进行随机亮度调整，使用tf.image.random_brightness()函数
        brighted_image = tf.image.random_brightness(flipped_image, max_delta=0.8)
        # 将亮度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数
        contrasted_image = tf.image.random_contrast(brighted_image, lower=0.2, upper=1.8)
        # 进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差
        standarded_image = tf.image.per_image_standardization(contrasted_image)

        # 设置图片数据及标签的形状
        standarded_image.set_shape([24,24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_example_per_epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)

        images_train,labels_train=tf.train.shuffle_batch([standarded_image,read_input.label],batch_size=batch_size,
                                                         num_threads=16,
                                                         capacity=min_queue_examples + 3 * batch_size,
                                                         min_after_dequeue=min_queue_examples,
                                                         )
                             #使用tf.train.shuffle_batch()函数随机产生一个batch的image和label

        return images_train,tf.reshape(labels_train,[batch_size])
    else:
        # 在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        float_image = tf.image.per_image_standardization(resized_image)
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_example_per_epoch * 0.4)  #这边取0.4是为啥

        images_test,labels_test=tf.train.batch([float_image,read_input.label],
                                              batch_size=batch_size,num_threads=16,
                                              capacity=min_queue_examples + 3 * batch_size)
                                 #这里使用batch()函数代替tf.train.shuffle_batch()函数
        return images_test,tf.reshape(labels_test,[batch_size])
