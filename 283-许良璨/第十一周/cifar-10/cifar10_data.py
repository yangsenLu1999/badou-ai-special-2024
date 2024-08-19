import os
import tensorflow as tf

num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

class cifar10_record():
    pass

def cifar10_dataget(filequeue):
    result=cifar10_record()
    label_bytes=1
    result.H=32
    result.W=32
    result.C=3
    img_bytes=result.H*result.W*result.C
    record_bytes= label_bytes+img_bytes

    reader=tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key,value=reader.read(filequeue)
    data_RAW=tf.decode_raw(value,tf.uint8)
    result.label=tf.cast(tf.strided_slice(data_RAW,[0],[label_bytes]),tf.int32)
    reshaped_img=tf.reshape(tf.cast(tf.strided_slice(data_RAW,[label_bytes],[record_bytes]),tf.float32),[result.C,result.H,result.W])
    result.img=tf.transpose(reshaped_img,[1,2,0])

    return result

def input(datadir,batch_size,distorted):
    filename=[os.path.join(datadir,"data_batch_%d.bin"%i) for i in range(1,6)]
    filequeue=tf.train.string_input_producer(filename)
    read_input=cifar10_dataget(filequeue)
    imput_img=read_input.img

    if distorted :
        cropped_img=tf.random_crop(imput_img,[24,24,3])
        flipped_img=tf.image.random_flip_left_right(cropped_img)
        just_brightness=tf.image.random_brightness(flipped_img,max_delta=0.8)
        just_contrast=tf.image.random_contrast(just_brightness,lower=0.2,upper=1.8)
        float_img=tf.image.per_image_standardization(just_contrast)

        float_img.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_pre_epoch_for_eval * 0.4)

        label_train, img_train=tf.train.shuffle_batch([read_input.label,float_img],
                                                          batch_size=batch_size,
                                                          num_threads=16,
                                                          capacity=min_queue_examples + 3 * batch_size,
                                                          min_after_dequeue=min_queue_examples,
                                                          )
        return tf.reshape(label_train,[batch_size]),img_train
    else:
        cropped_img=tf.image.resize_image_with_crop_or_pad(imput_img,24,24)
        float_img = tf.image.per_image_standardization(cropped_img)
        float_img.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_train * 0.4)

        label_test, img_test = tf.train.batch([read_input.label, float_img],
                                                            batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            )
        return tf.reshape(label_test,[batch_size]), img_test



