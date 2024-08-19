# 这段代码实现了一个从文件中生成图像数组的生成器函数，主要用于深度学习框架 Keras 的模型训练。
'''导入 Keras 的回调函数模块。这些功能用于在训练过程中监控和优化模型表现：
TensorBoard：可视化训练过程。
ModelCheckpoint：在每个训练周期结束时保存模型。
ReduceLROnPlateau：在指标停止改善时降低学习率。
EarlyStopping：在指标没有改善时提前停止训练'''
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.utils import np_utils
'''从 Keras 导入 np_utils，它提供了许多与 NumPy 有关的工具函数，例如将分类标签转换为 one-hot 编码'''
from keras.optimizers import Adam
'''导入 Adam 优化器，常用于神经网络训练，以自适应学习率优化模型'''
from model.AlexNet import AlexNet
'''从自定义模型模块中导入 AlexNet 类，表示要使用的网络结构'''
import numpy as np
import utils
import cv2
from keras import backend as K

K.set_image_dim_ordering('tf')
'''设置 Keras 的图像维度顺序，这里指定为 tf（TensorFlow），表示输入数据的格式为 (batch_size, height, width, channels)'''


def generate_arrays_from_file(lines, batch_size):
    '''定义生成器函数 generate_arrays_from_file，接受两个参数：
lines：包含文件行的列表，每行包含图像文件名及其对应标签，以 ; 分隔。
batch_size：每次生成的数据批量大小。'''
    # 获取总长度
    n = len(lines)
    i = 0
    # n：获取输入文件行的总数。
    # i：索引初始化为 0，用于循环访问行
    while 1:   # 启动一个无限循环，让生成器持续生成数据
        X_train = []
        Y_train = []    # 初始化用于存储图像数据和标签的空列表
        # 获取一个batch_size大小的数据
        for b in range(batch_size):   # 为每个批处理循环 batch_size 次来填充 X_train 和 Y_train。
            if i == 0:
                np.random.shuffle(lines)   # 当索引 i 为 0 时，对输入的 lines 列表进行随机打乱，以增加训练的多样性
            name = lines[i].split(';')[0]  # 从当前行中提取图像文件名，lines[i].split(';')[0] 获取 ; 前的部分
            # 从文件中读取图像
            img = cv2.imread(r".\data\image\train" + '/' + name)
            # 路径由 ".\data\image\train/" 和图像名 name 组合而成
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])   # 从当前行中提取标签，lines[i].split(';')[1] 获取 ; 后的部分，
            # 并将标签添加到 Y_train 列表中
            # 读完一个周期后重新开始,更新索引 i，以循环访问列表。当 i 达到总长度 n 时，重新开始
            i = (i + 1) % n
        # 处理图像
        X_train = utils.resize_image(X_train, (224, 224))   # utils 模块的 resize_image 函数
        X_train = X_train.reshape(-1, 224, 224, 3)   # 改变 X_train 的形状，确保它符合模型的输入要求。
        # 这将形成的形状为 (batch_size, 224, 224, 3
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
        # 将 Y_train 转换为 one-hot 编码。num_classes=2 表明有两个类别，适用于二分类问题
        yield (X_train, Y_train)  # 生成器返回当前批次的图像和标签，以便后续训练使用。此语句会在每次调用生成器时返回一组数据


if __name__ == "__main__":   #  是用于判断该脚本是否是被直接执行的模块。只有当脚本被直接运行时，下面的代码才会执行。
    # 模型保存的位置
    log_dir = "./logs/"   # 模型load进来,设定了模型训练过程中的日志保存路径，通常用于存储训练过程中的模型权重、日志文件等

    # 打开数据集的txt
    with open(r".\data\dataset.txt", "r") as f:
        lines = f.readlines()   # 将文件中的所有行读取到 lines 列表中。每一行通常代表一条数据的路径或标签

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)  # 通过 np.random.seed(10101) 设置随机数种子，以便在打乱数据时能够得到可重复的结果
    np.random.shuffle(lines)  # np.random.shuffle(lines) 将 lines 列表中的内容打乱顺序。随机打乱数据有助于模型更好地学习，避免因数据顺序造成偏见
    np.random.seed(None)  # np.random.seed(None) 取消固定的种子设置，使接下来的随机操作恢复为系统默认状态

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 建立AlexNet模型
    model = AlexNet()    # 这一行创建了一个名为 AlexNet 的模型实例。这个模型通常是指在定义 AlexNet 类的文件中实现的卷积神经网络结构
    # 换这一行，变成别的模型，下面几乎不用改，就能去训练，还有调参，
    # 或者在定义AlexNet文件里改模型结构，换成VGG之类的，train和predict文件直接用就行

    '''这段代码主要是用于设置训练过程中模型的回调函数、损失函数、优化器和训练相关参数'''
    # 保存的方式，3代保存一次
    checkpoint_period1 = ModelCheckpoint(  # ModelCheckpoint：这是 Keras 中用于在训练过程中监控模型性能并保存模型的回调函数
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        # 这个参数定义了保存模型的文件名格式。它将在每个指定的周期后保存模型.
        '''ep{epoch:03d}：当前 epoch 的编号（3 位数，不足的前面补零）。
loss{loss:.3f}：训练损失，保留三位小数。
val_loss{val_loss:.3f}：验证损失，保留三位小数'''
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )
    '''monitor='acc'：指定监控的指标为准确率（accuracy）。
save_weights_only=False：设置为 False 表示保存整个模型（包括结构和权重），而不仅仅是权重。
save_best_only=True：只有当监控的指标（此处为准确率）得到改善时，才会保存模型。
period=3：表示每隔 3 个 epochs 保存一次模型（在较新版本中，这个参数被更改为save_freq）。'''


    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )
    '''ReduceLROnPlateau：这是一个 Keras 回调，在监控的指标停止改善时自动降低学习率。
monitor='acc'：监控准确率。
factor=0.5：当监控指标在指定的 patience 次 epochs 内没有改善时，学习率将乘以 0.5（即减半）。
patience=3：表示在没有改善的情况下等待读取 3 个 epochs。
verbose=1：设置为 1 时，会在学习率下降时输出相关信息。'''


    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )
    '''EarlyStopping：用于停止训练以防止过拟合的回调函数。
monitor='val_loss'：监控验证损失。
min_delta=0：模型必须改善的最小幅度；如果改进幅度小于这个值，则视为没有改进。
patience=10：表示如果验证损失在 10 个 epochs 内没有改善，则停止训练。
verbose=1：设置为 1 时，会在早停触发时输出相关信息。'''

    # 交叉熵
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    '''model.compile()：用于配置模型。
loss='categorical_crossentropy'：损失函数，这里使用的是类别交叉熵，适合多类别分类问题。
optimizer=Adam(lr=1e-3)：使用 Adam 优化器，学习率设置为 0.001（1e-3）。
metrics=['accuracy']：训练和评估时监控准确率（accuracy）。'''

    # 一次的训练集大小
    batch_size = 128  # 设置每次训练中使用的样本批次大小为 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    # print(...)：输出训练和验证样本数量及批次大小。 num_train 是训练样本数量，num_val 是验证样本数量

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr])
    '''model.fit_generator()：这是 Keras 中用于训练模型的一种方法
    generate_arrays_from_file(lines[:num_train], batch_size)：这是一个自定义生成器函数，负责从指定的文件或数据结构中生成训练数据。
    它接收前 num_train 行数据（lines[:num_train]）并以 batch_size 为大小逐批生成数据
    steps_per_epoch：每个 epoch 中的步数，这里计算为 num_train（训练样本数）除以 batch_size。如果结果小于 1，则使用 1，
    以确保至少有一个步数（数据不会为空）。
    validation_data：用于验证的输入数据，同样使用自定义生成器从 lines 的剩余部分（即 lines[num_train:]）中生成验证数据
    validation_steps：每个 epoch 中用于验证的步数，计算为 num_val（验证样本数）除以 batch_size。如果结果小于 1，同样指定为 1
    epochs：指定要训练的总 epoch 数，这里设置为 50，即进行 50 次完整的训练轮数
    initial_epoch：指定从哪个 epoch 开始训练。如果是从头开始训练就设置为 0，如果是继续之前的训练，那么应该设置为之前的 epoch 编号
    callbacks：传入一个回调列表，这里包括 checkpoint_period1 和 reduce_lr，用于在训练过程中执行特定的操作。例如：
    checkpoint_period1 可能是保存模型状态的回调。
    reduce_lr 用于根据监控指标动态调整学习率
    '''
    model.save_weights(log_dir + 'last1.h5')
    '''model.save_weights(log_dir + 'last1.h5')：保存当前训练好的模型权重到指定的文件路径。
log_dir 是一个目录（路径），'last1.h5' 是文件名，合在一起形成了权重存储的完整路径（例如 /path/to/log_dir/last1.h5）。
该行代码的目的是将模型的权重保存下来，以便在后续使用或交付给甲方（客户）进行继续训练或推断'''

