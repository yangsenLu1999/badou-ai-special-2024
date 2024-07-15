from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K


# K.set_image_dim_ordering('tf')

# 从文件中读取数据，并以批量的形式生成训练数据和标签，以便进行深度学习模型的训练。
# 入参： lines（文件中的行列表）和 batch_size（批量大小）。
def generate_arrays_from_file(lines,batch_size):
    n = len(lines) # 获取文件的总行数 n
    i = 0 # 初始化一个指针 i 为 0
    while 1: # 无限循环
        X_train = []  # 用于存储当前批量的训练数据
        Y_train = []  # 用于存储当前批量的训练标签
        # 获取一个batch_size大小的数据  对于批量中的每个样本（从 0 到 batch_size-1）
        for b in range(batch_size):
            if i==0:  # 如果指针 i 为 0，对行列表进行随机洗牌，以确保数据的随机性
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]  # 根据行列表中的索引 i，获取文件名，并从文件中读取图像
            # 从文件中读取图像  将图像转换为 RGB 格式，并进行归一化处理（除以 255）
            img = cv2.imread(r".\data\image\train" + '/' + name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img/255
            X_train.append(img)  # 将处理后的图像添加到 X_train 列表中
            Y_train.append(lines[i].split(';')[1])  # 将行列表中索引 i 处的标签添加到 Y_train 列表中
            # 读完一个周期后重新开始  递增指针 i，并使用取模运算确保指针不会超出行列表的范围
            i = (i+1) % n
        # 处理图像 对图像进行处理，将其大小调整为 (224, 224)，并将其形状重塑为适合模型输入的格式。
        X_train = utils.resize_image(X_train,(224,224))
        X_train = X_train.reshape(-1,224,224,3)
        # 将标签进行独热编码，以便与模型的输出进行比较
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 2)
        # 使用 yield 关键字生成一个元组 (X_train, Y_train)，表示当前批量的训练数据和标签
        yield (X_train, Y_train)


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r".\data\dataset.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    '''
    打乱列表顺序：使用np.random.shuffle函数打乱列表lines的顺序。在打乱之前，设置了随机数种子为10101，以确保每次运行结果的可重复性。
               打乱完成后，将随机数种子设置为None，以恢复默认的随机数生成状态。 通过打乱数据的顺序，可以增加数据的随机性和多样性，避免模型过度依赖数据的顺序或出现过拟合的情况
    '''
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # 建立AlexNet模型
    model = AlexNet()
    
    # 保存的方式，3世代保存一次
    '''
    定义了一个 `ModelCheckpoint` 对象，用于在模型训练过程中进行检查点的保存。        
        - `log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'`：这是保存检查点文件的路径和文件名格式。`log_dir` 是日志目录，
                                                   `ep{epoch:03d}` 表示当前训练轮数，`loss{loss:.3f}` 表示当前训练损失，`val_loss{val_loss:.3f}` 表示当前验证损失。
        - `monitor='acc'`：指定要监控的指标，这里是准确率（`acc`）。
        - `save_weights_only=False`：表示是否只保存模型的权重。如果设置为 `True`，则只保存模型的权重；如果设置为 `False`，则保存整个模型。
        - `save_best_only=True`：表示是否只保存最优的检查点。如果设置为 `True`，则只有当监控指标的值优于之前保存的最优值时，才会保存检查点；如果设置为 `False`，则每次训练都会保存检查点。
        - `period=3`：表示每经过多少个训练轮数保存一次检查点。
        
        综上所述，这段代码的作用是在模型训练过程中，每经过 3 个训练轮数，保存一次检查点文件，文件名为当前训练轮数、训练损失、验证损失的组合，并根据准确率指标保存最优的检查点。这些检查点可以用于模型的恢复、继续训练或评估。
    '''
    checkpoint_period1 = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='acc', 
                                    save_weights_only=False, 
                                    save_best_only=True, 
                                    period=3
                                )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    '''
    定义了一个 `ReduceLROnPlateau` 对象，用于在模型训练过程中根据监控指标的值自动调整学习率。        
        - `monitor='acc'`：指定要监控的指标，这里是准确率（`acc`）。
        - `factor=0.5`：表示学习率衰减的因子。当监控指标在一定的轮数（由 `patience` 参数指定）内没有改善时，学习率将乘以这个因子。
        - `patience=3`：表示在学习率衰减之前，监控指标没有改善的轮数。
        - `verbose=1`：表示是否在控制台输出相关信息。
        综上所述，这段代码的作用是在模型训练过程中，根据准确率指标的值自动调整学习率。如果准确率在一定的轮数内没有改善，学习率将乘以 0.5。这样可以在训练过程中动态地调整学习率，以提高模型的性能。
    '''
    reduce_lr = ReduceLROnPlateau(
                            monitor='acc', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    '''
    用于在模型训练过程中实现早停（Early Stopping）
        `EarlyStopping` 是一种在训练过程中监控验证集上的指标，并在指标不再改善时提前停止训练的技术。这样可以避免过度训练，节省计算资源，并提高模型的泛化能力。        
        - `monitor='val_loss'`：指定要监控的指标。在这里，我们选择监控验证集上的损失（`val_loss`）。
        - `min_delta=0`：表示当监控指标的改善小于 `min_delta` 时，不被认为是真正的改善。将其设置为 0 意味着只要验证损失不再下降，就会触发早停。
        - `patience=10`：指定在验证损失不再改善的情况下，继续训练的轮数。在这个例子中，如果验证损失在连续 10 轮训练中都没有改善，训练将被提前停止。
        - `verbose=1`：控制是否在控制台输出早停的相关信息。设置为 1 表示输出详细信息。
        通过使用 `EarlyStopping`，可以在模型训练过程中自动检测验证集上的性能不再提高的情况，并及时停止训练，避免浪费时间和资源在不必要的训练上。这有助于找到模型的最佳训练轮数，并提高模型的性能和泛化能力。
    '''
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=1
                        )

    # 交叉熵
    '''
    使用`compile`方法来配置模型的训练过程。
        - `loss = 'categorical_crossentropy'`：指定模型的损失函数。在多类别分类问题中，通常使用`categorical_crossentropy`作为损失函数。它衡量了模型预测的类别概率分布与真实类别分布之间的差异。
        - `optimizer = Adam(lr=1e-3)`：指定模型的优化器。在这里，我们选择使用`Adam`优化器，它是一种常用的优化算法。`lr=1e-3`表示学习率，控制着模型参数更新的步长。
        - `metrics = ['accuracy']`：指定模型在训练和评估过程中要计算的指标。在这里，我们选择计算准确率（`accuracy`）作为评估指标。
        通过配置这些参数，模型将使用`categorical_crossentropy`损失函数进行优化，并使用`Adam`优化器来更新模型参数。在训练过程中，模型将计算准确率作为评估指标，以便评估模型的性能。
    '''
    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr=1e-3),
            metrics = ['accuracy'])

    # 一次的训练集大小
    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # 开始训练  存储训练的模型
    '''
    使用`fit_generator`方法来训练模型。        
        - `generate_arrays_from_file(lines[:num_train], batch_size)`：这是一个生成器函数，用于生成训练数据的批量。`lines[:num_train]`表示训练数据的行数，`batch_size`表示每个批量的大小。
        - `steps_per_epoch=max(1, num_train//batch_size)`：指定每个训练轮数的步数。在这里，我们将其设置为训练数据的数量除以批量大小，并向上取整。这样可以确保每个训练轮数都能遍历完所有的训练数据。
        - `validation_data=generate_arrays_from_file(lines[num_train:], batch_size)`：这是一个生成器函数，用于生成验证数据的批量。`lines[num_train:]`表示验证数据的行数，`batch_size`表示每个批量的大小。
        - `validation_steps=max(1, num_val//batch_size)`：指定每个验证轮数的步数。在这里，我们将其设置为验证数据的数量除以批量大小，并向上取整。这样可以确保每个验证轮数都能遍历完所有的验证数据。
        - `epochs=50`：指定训练的轮数。
        - `initial_epoch=0`：指定初始的训练轮数。
        - `callbacks=[checkpoint_period1, reduce_lr]`：指定训练过程中的回调函数。在这里，我们使用了两个回调函数：`checkpoint_period1`用于在每个训练轮数结束时保存模型的检查点；`reduce_lr`用于在验证损失不再改善时降低学习率。
        最后，代码使用`save_weights`方法来保存模型的权重。
        总的来说，这段代码的作用是使用生成器函数来生成训练数据和验证数据的批量，并使用`fit_generator`方法来训练模型。在训练过程中，模型将使用`categorical_crossentropy`损失函数进行优化，
                并使用`Adam`优化器来更新模型参数。在训练过程中，模型将计算准确率作为评估指标，以便评估模型的性能。
    '''
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[checkpoint_period1, reduce_lr])
    model.save_weights(log_dir+'last1.h5')

