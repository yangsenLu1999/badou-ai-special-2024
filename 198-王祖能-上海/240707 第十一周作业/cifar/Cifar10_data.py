'''
文件对cifar10数据并进行预处理，以tensorflow框架为主
60000张32*32的彩色图像，这些图像共分为10个类别，每个类别都有6000张图像
'''
import tensorflow as tf

num_classes = 10  # cifar10共10种分类
num_per_epoch_for_train = 50000
num_per_epoch_for_test = 10000  # 定义训练和测试样本个数


class Cifar10Record(object):  # 空类用于返回读取的cifar10数据
    pass


def read_cifar10(file_queue):  # 读取文件内容，固定长度字符串转为图像格式，并分割标签、带通道图片数据，作为带方法的对象结果输出
    result = Cifar10Record()
    result.h = 32
    result.w = 32
    result.c = 3
    labels_bytes = 1   # cifar10表示0~9需要1个占位，cifar100需要2个占位
    img_bytes = result.h * result.w * result.c  # 图像占用的总像素数
    data_bytes = labels_bytes + img_bytes  # 总数据 =标签数据+图像数据

    reader = tf.FixedLengthRecordReader(record_bytes=data_bytes)  # 固定长度的文件读取器，转化为records，每个records包含key, value对
    result.key, value = reader.read(file_queue)  # 其中value包含了label和img
    result.all = tf.decode_raw(value, tf.uint8)  # 读取的字符串类型张量变为图片的uint8格式
    result.label_init = tf.strided_slice(result.all, [0], [labels_bytes])  # 读取labelbytess位标签, start和end必须带中括号[]
    result.img_init = tf.strided_slice(result.all, [labels_bytes], [data_bytes])  # 读取其余位作为图片数据
    '''tf.strided_slice()多维切片
    input = tf.constant([[[5, 2, 3], [2, 33, 1]],
                         [[2, 3, 2], [0, 5, 7]],
                         [[1, 10, 3], [6, 12, 20]]])
    tf.strided_slice(input, [1, 0, 0], [2, 1, 3], [1, 1, 1])
    左闭右开，第一维取1:2，即[[2, 3, 2], [0 ,5, 7]]，第二维取0：1即[2, 3, 2], 第三维取0:3即全部，[2, 3, 2]，步长为1
    '''
    result.label = tf.cast(result.label_init, tf.int32)  # 标签格式转换，其实不转换也是足够的
    result.img_new = tf.reshape(result.img_init, [result.c, result.h, result.w])  # tf而言[h, w, c]和[c, h , w]都可以，reshape张量维度的改变
    result.img = tf.transpose(result.img_new, [1, 2, 0])  # 张量维度顺序的变化，原来第0位变为第三位，即[h, w, c]
    return result  # 输出结果中最重要的就是result.img 和result.label


def inputs(data_dir, batch_size, distorted):  # 读入数据，图像预处理，是否进行图像增强处理，增加样本数量
    import os
    filepath = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]  # 文件下5个bin拼合在一起
    file_queue = tf.train.string_input_producer(filepath)
    # 创建文件队列，string_tensor传入张量或文件名列表，读取线程源源不断地将文件系统中的图片读入到内存队列，负责计算是另一个线程，需要数据时，直接从内存队列中取
    data = read_cifar10(file_queue)  # 读取上一个函数，把队列中的数据分割处理提取labels和features
    img = tf.cast(data.img, tf.float32)  # 为了精度转换输入图片深度
    num_per_epoch = num_per_epoch_for_train

    if distorted:  # 在训练过程中，需要True进行图像增强，以下为随机位置进行处理，也可以不用random图像通篇加强
        cropped_img = tf.random_crop(img, [24, 24, 3])  # 随机切割固定尺寸大小的图像
        flipped_img = tf.image.random_flip_left_right(cropped_img)  # 随机左右翻转，也可以上下翻
        bright_img = tf.image.random_brightness(flipped_img, max_delta=0.8)  # 随机位置进行亮度增强
        contrast_img = tf.image.random_contrast(bright_img, lower=0.4, upper=1.6)  # 随机位置进行对比度增强,随机对比因子的上下限
        normal_img = tf.image.per_image_standardization(contrast_img)  # 数据标准化，(x-mu)/sigma，范围变了分布不变，减小不同尺度的不利影响
        normal_img.set_shape([24, 24, 3])  # 冗余，只为了增加函数内容
        data.label.set_shape([1])

        min_queue_batch = int(0.4 * num_per_epoch_for_test)  # 为什么用推理的数据集数量得到batch数????考虑需要进行训练和测试，以数量较小的估计，不会溢出，79行用的又是train的数量
        print('Filling queue with %d Cifar images before starting to train. This will take a few minutes.' % min_queue_batch)
        img_train, label_train = tf.train.shuffle_batch([normal_img, data.label], batch_size=batch_size, num_threads=16,
                                                        capacity=min_queue_batch + 3 * batch_size, min_after_dequeue=min_queue_batch,)
        '''
        ！！只取出一个batch,而不是所有的都取出来了
        传入张量组成的随机乱序队列,随机产生一个batch的image和label. 相当于min + 3batch之后不断从中取出batch, 最后不少于min, 充分摇匀数据
        capacity:整数,队列中的最大的元素数，一定要比min_after_dequeue大,决定了可以进行预处理操作元素的最大值.
        min_after_dequeue: 当一次出列操作完成后,队列中元素的最小数量,往往用于定义元素的混合级别,定义了随机取样的缓冲区大小,此参数越大表示更大级别的混合但是会导致启动更加缓慢,并且会占用更多的内存
        batch_size进行一次批处理的tensors数量.
        num_threads大于1,使用多个线程在tensor_list中读取文件
        '''
        return img_train, tf.reshape(label_train, [batch_size])  # 这里为什么要用reshape变形?也可以不用
    else:  # 推理过程中不需要图像处理，False
        resized_img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)  # 原始尺寸较大裁取正中该尺寸，原始尺寸较小采用空白0填充
        normal_img = tf.image.per_image_standardization(resized_img)
        normal_img.set_shape([24, 24, 3])
        data.label.set_shape([1])

        min_queue_batch = int(0.4 * num_per_epoch)  # 跟第51行，相当于用了train的num, 按个人喜好设置的，不具有标准性
        img_test, label_test = tf.train.shuffle_batch([normal_img, data.label], batch_size=batch_size, num_threads=16,
                                                        capacity=min_queue_batch + 3 * batch_size, min_after_dequeue=min_queue_batch)
        return img_test, tf.reshape(label_test, [batch_size])
