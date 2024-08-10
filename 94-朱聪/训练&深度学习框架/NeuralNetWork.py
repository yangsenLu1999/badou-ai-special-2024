import scipy.special
import numpy


class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningrate
        '''
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        '''
        # self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5 # 生成3*3的二维数组
        # self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        '''
        scipy.special.expit对应的是sigmod函数.
        lambda是Python关键字，类似C语言中的宏定义，当我们调用self.activation_function(x)时，编译器会把其转换为spicy.special_expit(x)。
        '''
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self):
        '''
        [1]
        训练过程分两步：
        第一步是计算输入训练数据，给出网络的计算结果，这点跟我们前面实现的query()功能很像。正向反馈 wx+b -> 激活函数
        第二步是将计算结果与正确结果相比对，获取误差，采用误差反向传播法更新网络里的每条链路权重。

        我们先用代码完成第一步.

        inputs_list:输入的训练数据;
        targets_list:训练数据对应的正确结果。
        '''

        def train(self, inputs_list, targets_list):
            # 根据输入的训练数据更新节点链路权重
            '''
            把inputs_list, targets_list转换成numpy支持的二维矩阵
            .T表示做矩阵的转置
            '''
            inputs = numpy.array(inputs_list, ndmin=2).T # 指定生成的数组至少有 2 维
            targets = numpy.array(targets_list, nmin=2).T
            # 计算信号经过输入层后产生的信号量
            hidden_inputs = numpy.dot(self.wih, inputs)
            # 中间层神经元对输入的信号做激活函数后得到输出信号
            hidden_outputs = self.activation_function(hidden_inputs)
            # 输出层接收来自中间层的信号量
            final_inputs = numpy.dot(self.who, hidden_outputs)
            # 输出层对信号量进行激活函数后得到最终输出信号
            final_outputs = self.activation_function(final_inputs)


        '''
        [2]
        上面代码根据输入数据计算出结果后，我们先要获得计算误差.
        误差就是用正确结果减去网络的计算结果。
        在代码中对应的就是(targets - final_outputs).
        '''

        def train(self, inputs_list, targets_list):
            # 根据输入的训练数据更新节点链路权重
            '''
            把inputs_list, targets_list转换成numpy支持的二维矩阵
            .T表示做矩阵的转置
            '''
            inputs = numpy.array(inputs_list, ndmin=2).T # 指定生成的数组至少有 2 维
            targets = numpy.array(targets_list, nmin=2).T
            # 计算信号经过输入层后产生的信号量
            hidden_inputs = numpy.dot(self.wih, inputs)
            # 中间层神经元对输入的信号做激活函数后得到输出信号
            hidden_outputs = self.activation_function(hidden_inputs)
            # 输出层接收来自中间层的信号量
            final_inputs = numpy.dot(self.who, hidden_outputs)
            # 输出层对信号量进行激活函数后得到最终输出信号
            final_outputs = self.activation_function(final_inputs)

            # 计算误差
            output_errors = targets - final_outputs
            hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
            # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上。最复杂的一部分，也是有固定公式的
            self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                            numpy.transpose(hidden_outputs))
            self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                            numpy.transpose(inputs))
            pass


        '''
        [3]
        使用实际数据来训练我们的神经网络
        '''
        # open函数里的路径根据数据存储的路径来设定
        # data_file = open("dataset/mnist_test.csv")
        # data_list = data_file.readlines()
        # data_file.close()
        # len(data_list)
        # data_list[0]
        '''
        这里我们可以利用画图.py将输入绘制出来
        '''


        '''
        [4]
        从绘制的结果看，数据代表的确实是一个黑白图片的手写数字。
        数据读取完毕后，我们再对数据格式做些调整，以便输入到神经网络中进行分析。
        我们需要做的是将数据“归一化”，也就是把所有数值全部转换到0.01到1.0之间。
        由于表示图片的二维数组中，每个数大小不超过255，由此我们只要把所有数组除以255，就能让数据全部落入到0和1之间。
        有些数值很小，除以255后会变为0，这样会导致链路权重更新出问题。
        所以我们需要把除以255后的结果先乘以0.99，然后再加上0.01，这样所有数据就处于0.01到1之间。
        '''
        scaled_input = image_array / 255.0 * 0.99 + 0.01
        # 根据输入的训练数据更新节点链路权重
        pass

    def query(self, inputs):
        # 根据输入数据计算并输出答案。
        # 计算中间层从输入层接收到的信号量。推理的过程就是前向传播。训练是前向+反向
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


'''
我们尝试传入一些数据，让神经网络输出结果试试.
程序当前运行结果并没有太大意义，但是至少表明，我们到目前为止写下的代码没有太大问题，
'''
# input_nodes = 3
# hidden_nodes = 3
# output_nodes = 3
#
# learning_rate = 0.3
# n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# n.query([1.0, 0.5, -1.5])


# 初始化网络
'''
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
'''
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读入训练数据
# open函数里的路径根据数据存储的路径来设定
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 加入epochs,设定网络的训练循环次数
epochs = 5
for e in range(epochs):
    # 把数据依靠','区分，并分别读入。取到的每条数据第一个值，就是这张图片代表的数字
    for record in training_data_list:
        all_values = record.split(',')
        # 归一化
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01  # numpy.asfarray,将输入转换为浮点数类型的NumPy数组
        # 设置图片与数值的对应关系
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99 # one hot转换
        n.train(inputs, targets) # 训练好的模型是指，已经通过大量的训练得到了最后的权重值等信息，基于这些值去处理测试的输入

test_data_file = open("mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    # 预处理数字图片
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字
    outputs = n.query(inputs) # 测试的过程是正向传播的，训练的时候才需要不断通过反向传播来对权重值进行优化
    # 找到数值最大的神经元对应的编号
    label = numpy.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)
