# 该文件的目的是构造神经网络的整体结构，并进行训练和测试（评估）过程
# 和tf过程差不多, 主要在于卷积部分
import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data

max_steps = 4000
batch_size = 100
num_examples_for_eval = 10000
data_dir = "Cifar_data/cifar-10-batches-bin"


# 创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weights_loss")
        tf.add_to_collection("losses", weights_loss)
    return var


# 使用上一个文件里面已经定义好的文件序列读取函数读取训练数据文件和测试数据从文件.
# 其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
# images_train  <tf.Tensor 'shuffle_batch:0' shape=(100, 24, 24, 3) dtype=float32>    labels_train <tf.Tensor 'Reshape_1:0' shape=(100,) dtype=int32>
images_train, labels_train = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
images_test, labels_test = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

# 创建x和y_两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值。
# 要注意的是，由于以后定义全连接网络的时候用到了batch_size，所以x中，第一个参数不应该是None，而应该是batch_size
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])  # <tf.Tensor 'Placeholder:0' shape=(100, 24, 24, 3) dtype=float32>
y_ = tf.placeholder(tf.int32, [batch_size])  # <tf.Tensor 'Placeholder_1:0' shape=(100,) dtype=int32>

# 创建第一个卷积层 shape=(kh,kw,ci,co)
'''
定义一个卷积神经网络（CNN）的卷积层和池化层, 并对输入图像进行特征提取和降维,以便后续的处理和分类
    1. `kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)`：
                创建了一个卷积核 `kernel1`，它的形状为 `[5, 5, 3, 64]`。其中，`5` 和 `5` 分别表示卷积核的高度和宽度，`3` 表示输入图像的通道数，`64` 表示卷积核的数量。
                `stddev=5e-2` 表示卷积核的标准差，`w1=0.0` 是一个可选的权重衰减参数。
    2. `conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")`：
                执行卷积操作: `x` 是输入图像，`kernel1` 是卷积核，`[1, 1, 1, 1]` 表示卷积核的移动步长，
                `padding="SAME"` 表示使用相同的填充方式。卷积操作将输入图像与卷积核进行卷积，得到卷积后的特征图 `conv1`。
                                 卷积之后输出的feature map尺寸保持不变(相对于输入图片)。当然，same模式不代表完全输入输出尺寸一样，也跟卷积核的步长有关系。
                                 same模式也是最常见的模式，因为这种模式可以在卷积过程中让图的大小保持不变。
    3. `bias1 = tf.Variable(tf.constant(0.0, shape=[64]))`：
                创建了一个偏置项 `bias1`，它的形状为 `[64]`，与卷积核的数量相同。
    4. `relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))`：
                对卷积后的特征图进行激活函数处理: `tf.nn.relu` 是 ReLU 激活函数，它将输入的负值置为 0，正值保持不变。`tf.nn.bias_add` 是将偏置项加到卷积后的特征图上。
    5. `pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")`：
                执行池化操作:`relu1` 是激活后的特征图，`ksize=[1, 3, 3, 1]` 表示池化窗口的大小，`strides=[1, 2, 2, 1]` 表示池化窗口的移动步长，`padding="SAME"` 表示使用相同的填充方式。
                          池化操作将特征图划分为不同的区域，然后在每个区域内取最大值，得到池化后的特征图 `pool1`。
                - `relu1`：这是输入的张量，通常是经过 ReLU 激活函数处理后的结果。
                - `ksize`：这是一个列表，指定了池化窗口的大小。在这个例子中，`ksize=[1, 3, 3, 1]` 表示池化窗口的高度为 1，宽度为 3，深度为 3。
                - `strides`：这也是一个列表，指定了池化操作的步长。在这个例子中，`strides=[1, 2, 2, 1]` 表示在高度和深度方向上的步长为 1，在宽度方向上的步长为 2。
                - `padding`：这是一个字符串，指定了池化操作的填充方式。在这个例子中，`padding="SAME"` 表示使用相同的填充方式，即在输入张量的边缘添加零，以确保输出张量的大小与输入张量相同。
                最大池化操作的作用是对输入张量进行下采样，减少张量的维度，同时保留输入张量中的重要特征。通过设置池化窗口的大小和步长，可以控制下采样的程度。
                在这个例子中，宽度方向上的步长为 2，意味着输出张量的宽度是输入张量的一半。
'''
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)  # <tf.Variable 'Variable:0' shape=(5, 5, 3, 64) dtype=float32_ref>
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")  # 四个方向 1111  <tf.Tensor 'Conv2D:0' shape=(100, 24, 24, 64) dtype=float32>
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))  # <tf.Variable 'Variable_1:0' shape=(64,) dtype=float32_ref>
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))  # <tf.Tensor 'Relu:0' shape=(100, 24, 24, 64) dtype=float32>
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")  # <tf.Tensor 'MaxPool:0' shape=(100, 12, 12, 64) dtype=float32>

# 创建第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)  # <tf.Variable 'Variable_2:0' shape=(5, 5, 64, 64) dtype=float32_ref>
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")  # <tf.Tensor 'Conv2D_1:0' shape=(100, 12, 12, 64) dtype=float32>
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))  # <tf.Variable 'Variable_3:0' shape=(64,) dtype=float32_ref>
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))  # <tf.Tensor 'Relu_1:0' shape=(100, 12, 12, 64) dtype=float32>
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")  # <tf.Tensor 'MaxPool_1:0' shape=(100, 6, 6, 64) dtype=float32>

'''
使用 TensorFlow 库进行张量的形状重塑和维度获取:
    1. `reshape = tf.reshape(pool2, [batch_size, -1])`：
                将张量 `pool2` 重塑为形状为 `[batch_size, -1]` 的张量。 其中，`batch_size` 是已知的批量大小，`-1` 表示自动计算该维度的大小，以使重塑后的张量 元素数量 与原始张量相同。
    2. `dim = reshape.get_shape()[1].value`：
                获取重塑后的张量 `reshape` 的形状信息，并从中获取第二个维度的值。`get_shape()[1].value` 表示获取形状信息中第二个维度的值，并将其存储在变量 `dim` 中。
    通过 `dim` 获取重塑后的张量的第二个维度的大小，以便在后续的计算中使用。
'''
# 因为要进行全连接层的操作，所以这里使用tf.reshape()函数将pool2输出变成一维向量，并使用get_shape()函数获取扁平化之后的长度
reshape = tf.reshape(pool2, [batch_size, -1])  # 这里面的-1代表将pool2的三维结构拉直为一维结构  <tf.Tensor 'Reshape_4:0' shape=(100, 2304) dtype=float32>
dim = reshape.get_shape()[1].value  # get_shape()[1].value表示获取reshape之后的第二个维度的值  dim 2304

'''
神经网络中添加一个全连接层： 将输入的维度 `dim` 映射到输出的维度 `384`。
    1. `weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)`: 
                 定义了一个权重变量 `weight1`，它的形状为 `[dim, 384]`，其中 `dim` 是上一步中计算得到的维度，`384` 是输出的维度。`stddev=0.04` 是权重的标准差，`w1=0.004` 是权重的初始值。
    2. `fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))`: 
                 定义了一个偏置变量 `fc_bias1`，它的形状为 `[384]`，其中 `384` 是输出的维度。
    3. `fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)`: 
               执行了一个全连接层的计算。首先，通过矩阵乘法将输入 `reshape` 和权重 `weight1` 相乘，然后将结果与偏置 `fc_bias1` 相加。最后，使用 ReLU 激活函数对结果进行非线性变换。    
'''
# 建立第一个全连接层
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)  # <tf.Variable 'Variable_4:0' shape=(2304, 384) dtype=float32_ref>
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))  # <tf.Variable 'Variable_5:0' shape=(384,) dtype=float32_ref>
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)  # <tf.Tensor 'Relu_2:0' shape=(100, 384) dtype=float32>

# 建立第二个全连接层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 建立第三个全连接层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)  # <tf.Variable 'Variable_8:0' shape=(192, 10) dtype=float32_ref>
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))  # <tf.Variable 'Variable_9:0' shape=(10,) dtype=float32_ref>
result = tf.add(tf.matmul(local4, weight3), fc_bias3)  # <tf.Tensor 'Add_2:0' shape=(100, 10) dtype=float32>

'''
计算交叉熵损失： 计算模型的预测结果与真实标签之间的交叉熵损失，并将结果存储在变量 `cross_entropy` 中。这个损失值将在后续的训练过程中用于优化模型的参数。
 `tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))`：这是 TensorFlow 提供的一个函数，用于计算稀疏标签的 softmax 交叉熵损失。
    - `logits`：这是模型的预测结果，通常是一个张量，表示每个类别对应的得分或概率。
    - `labels`：这是真实的标签，通常是一个整数张量，表示每个样本的类别标签。
    - `tf.cast(y_, tf.int64)`：这行代码将标签 `y_` 转换为整数类型，以与 `labels` 的类型匹配。
    交叉熵损失用于衡量模型的预测结果与真实标签之间的差异。它在深度学习中常用于分类问题，通过最小化交叉熵损失来优化模型的参数，以使模型能够更准确地预测类别。
    
'''
# 计算损失，包括权重参数的正则化损失和交叉熵损失  <tf.Tensor 'SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0' shape=(100,) dtype=float32>
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))

'''
使用 TensorFlow 库计算模型的正则化损失:  常用于在训练过程中对模型的参数进行正则化，以避免过拟合。正则化损失可以根据具体的应用和需求进行调整和优化。
    `tf.add_n(tf.get_collection("losses"))`：这行代码将 `tf.get_collection("losses")` 中收集的所有损失值相加。
       `tf.get_collection("losses")` 用于获取一个集合中存储的所有损失值。在 TensorFlow 中，可以通过将损失值添加到这个集合中来跟踪它们。
    通过将集合中的损失值相加，可以得到模型的总正则化损失。
'''
weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))  # <tf.Tensor 'AddN:0' shape=() dtype=float32>
'''
将交叉熵损失和正则化损失相加，并取平均值，得到最终的损失值。
    1. `tf.reduce_mean(cross_entropy)`：这行代码计算交叉熵损失的平均值。`tf.reduce_mean` 函数用于计算张量的平均值。
    2. `+ weights_with_l2_loss`：这行代码将正则化损失加到交叉熵损失的平均值上。
    通过将交叉熵损失和正则化损失相加，并取平均值，可以得到一个综合考虑了模型预测准确性和正则化项的损失值。这个损失值将在训练过程中用于优化模型的参数，以最小化损失并提高模型的性能。
    最终的损失值将用于评估模型的性能，并通过反向传播算法来更新模型的参数，以逐步提高模型的准确性和泛化能力。
'''
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss  # <tf.Tensor 'add_3:0' shape=() dtype=float32>
'''
使用 TensorFlow 中的 Adam 优化器来最小化损失函数 `loss`。
        1. `tf.train.AdamOptimizer(1e-3)`：创建一个 Adam 优化器对象。`1e-3` 是学习率（learning rate），用于控制每次参数更新的步长。
        2. `.minimize(loss)`：调用优化器的 `minimize` 方法，将损失函数作为参数传递进去。这告诉优化器要通过调整模型的参数来最小化损失函数。
    优化器的作用是根据损失函数的梯度信息，更新模型的参数，以使模型在训练数据上的损失最小化。Adam 优化器是一种常用的优化算法，它结合了自适应学习率和动量的概念，能够有效地进行参数优化。
           当执行 `train_op` 时，优化器会根据当前的参数值和损失函数的梯度，计算出参数的更新量，并将其应用到模型中。通过不断地迭代训练，模型的参数会逐渐调整，以提高模型的性能。  
    学习率是一个重要的超参数，需要根据具体的问题和数据集进行调整。如果学习率过大，可能会导致模型不收敛或过拟合；如果学习率过小，可能会导致训练速度缓慢。此外，还可以根据需要进行其他的优化策略和超参数调整，以获得更好的训练效果。
'''
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)  # <tf.Operation 'Adam' type=NoOp>

'''
使用 TensorFlow 库中的 `tf.nn.in_top_k` 函数来判断模型的预测结果是否在真实标签的前 `k` 个预测结果中。

`tf.nn.in_top_k(result, y_, 1)`：
    - `result`：这是模型的预测结果，通常是一个张量，表示每个样本的预测类别得分或概率。
    - `y_`：这是真实的标签，通常是一个整数张量，表示每个样本的真实类别。
    - `1`：这是要检查的前 `k` 个预测结果。在这里，`1` 表示只检查前一个预测结果。
函数的返回值是一个布尔型张量，表示每个样本的预测结果是否在前 `k` 个预测结果中。如果预测结果在真实标签的前 `k` 个预测结果中，则返回 `True`，否则返回 `False`。
这个操作通常用于评估模型的性能，特别是在多类别分类问题中。通过计算预测结果在前 `k` 个中的准确率，可以了解模型对不同类别的预测能力。
例如，如果 `top_k_op` 的结果为 `[True, False, True, False]`，则表示第一个和第三个样本的预测结果在前 `k` 个预测结果中，而第二个和第四个样本的预测结果不在前 `k` 个中。
你可以根据需要调整 `k` 的值来评估模型在不同前 `k` 个预测结果中的性能。
'''
# 函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
top_k_op = tf.nn.in_top_k(result, y_, 1)  # <tf.Tensor 'in_top_k/InTopKV2:0' shape=(100,) dtype=bool>

'''
初始化所有的全局变量:
在 TensorFlow 中，当定义了变量（如权重和偏置）后，它们的初始值通常是随机的或需要通过训练来学习。在训练之前，需要对这些变量进行初始化，以便后续的计算和优化。
`tf.global_variables_initializer` 函数会遍历所有的全局变量，并将它们初始化为默认值或根据变量的定义进行初始化。
当执行 `init_op` 时，TensorFlow 会执行初始化操作，将所有的全局变量设置为初始值。
* 需要注意的是，在执行训练或其他操作之前，必须先执行 `init_op` 来初始化变量。否则，可能会导致计算结果不准确或出现错误。

'''
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    # 变量初始化:  初始化了所有的全局变量, 在训练神经网络之前，需要先初始化变量，以便在训练过程中对其进行更新。
    sess.run(init_op)
    '''
    启动输入数据队列的线程:
        在 TensorFlow 中，数据通常以队列的形式进行输入。通过使用队列，可以实现数据的异步读取和预处理，提高数据的读取效率和训练的并行性。
        `tf.train.start_queue_runners` 函数会启动队列的线程，这些线程会从队列中读取数据，并将其提供给后续的计算操作。
        当执行这行代码时，它会启动所有与当前会话相关的输入数据队列的线程。这些线程会在后台运行，不断地从队列中读取数据，并将其传递给模型进行训练或推理。
        
        需要注意的是，在执行 `tf.train.start_queue_runners` 之前，必须已经创建了输入数据队列，并将数据放入队列中。否则，启动线程后可能会无法获取到数据。
        此外，还可以根据需要对队列进行更详细的配置和管理，例如设置队列的容量、数据的预处理操作等。
    '''
    # 启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()

    # 每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range(max_steps):
        start_time = time.time()
        # 在会话中运行 images_train 和 labels_train 操作，获取一批图像数据和对应的标签。
        image_batch, label_batch = sess.run([images_train, labels_train])
        # 在会话中运行 train_op 和 loss 操作，并将 image_batch 和 label_batch 作为输入数据。_ 表示忽略第一个返回值（通常是训练操作的输出），只获取损失值 loss_value
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
        # 计算训练的时间消耗
        duration = time.time() - start_time

        if step % 100 == 0:  # 每 100 步打印一次训练信息。
            examples_per_sec = batch_size / duration  # 计算每秒处理的样本数量。
            sec_per_batch = float(duration)  # 计算每批数据的处理时间
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (
            step, loss_value, examples_per_sec, sec_per_batch))  # 打印训练信息，包括当前步骤、损失值、每秒处理的样本数量和每批数据的处理时间

    '''
    在进行评估或测试时的一些计算和初始化操作：
        1. `num_batch = int(math.ceil(num_examples_for_eval / batch_size))`：计算评估或测试所需的批次数。使用 `math.ceil()` 函数向上取整，以确保批次数足够覆盖所有的评估样本。
        2. `true_count = 0`：初始化正确预测的样本数量为 0。
        3. `total_sample_count = num_batch * batch_size`：计算总的样本数量，即批次数乘以每批的样本数量。
        这些计算和初始化操作通常在评估或测试过程中使用，用于确定要处理的批次数、统计正确预测的样本数量以及计算评估指标等。
    '''
    # 计算最终的正确率
    num_batch = int(math.ceil(num_examples_for_eval / batch_size))  # math.ceil()函数用于求整
    true_count = 0
    total_sample_count = num_batch * batch_size

    '''
    循环，用于在评估或测试过程中处理多个批次的数据：
        1. `for j in range(num_batch)`：循环遍历评估或测试所需的批次数。
        2. `image_batch, label_batch = sess.run([images_test, labels_test])`：
                 在会话中运行 `images_test` 和 `labels_test` 操作，获取一批图像数据和对应的标签。
        3. `predictions = sess.run([top_k_op], feed_dict={x: image_batch, y_: label_batch})`：
                 在会话中运行 `top_k_op` 操作，根据输入的图像数据和标签，获取预测结果。预测结果通常是一个布尔值，表示预测是否正确。
        4. `true_count += np.sum(predictions)`：将预测结果中的正确预测数量累加到 `true_count` 中。`np.sum()` 函数用于计算数组中所有元素的总和。
        这段代码的作用是在评估或测试过程中，对多个批次的数据进行处理，并统计正确预测的数量。通过循环遍历批次数，获取图像数据和标签，进行预测，并将正确预测的数量累加到 `true_count` 中。
        最终，`true_count` 表示正确预测的总数量。        
    '''
    # 在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch, y_: label_batch})
        true_count += np.sum(predictions)

    # 打印正确率信息
    print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
