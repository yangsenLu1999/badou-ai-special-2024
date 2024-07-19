import torch


class Linear(torch.nn.Module):
    """
    通常将需要训练的层写在 init 函数中 (卷积、全连接 ...)
    定义了一个名为 `Linear` 的类，用于实现线性变换(神经网络中的前向传播和反向传播)。
        1. 在创建对象时进行初始化操作, 接收三个参数：`in_features` 表示输入特征的数量，`out_features` 表示输出特征的数量，`bias` 是一个布尔值，表示是否使用偏置项。
        2. 在方法内部，首先调用父类的 `__init__` 方法:
              在 Python 中，当我们定义一个类时，如果没有明确指定父类，那么默认的父类是 `object` 类。在这个例子中，`Linear` 类继承自 `torch.nn.Module` 类。
              调用父类的 `__init__` 方法的主要作用是确保父类的属性和方法被正确初始化。
                       在这个例子中，`torch.nn.Module` 类的 `__init__` 方法可能会执行一些与模块管理、参数注册等相关的操作，这些操作对于神经网络的正常运行是非常重要的。
              通过调用父类的 `__init__` 方法，可以确保 `Linear` 类继承了父类的所有属性和方法，并在初始化过程中进行了正确的设置。
                       这样就可以在 `Linear` 类中使用父类提供的功能，同时也可以根据需要添加自己的定制化逻辑。
        3. 然后，通过 `torch.nn.Parameter` 创建一个可学习的参数 `weight`，它是一个张量，形状为 `(out_features, in_features)`，并使用正态分布进行初始化。
              `torch.randn(x)` 是 PyTorch 库中的一个函数，用于生成一个具有标准正态分布的随机张量。
                    1. `torch.randn()` 函数接受一个参数，表示生成的张量的维度。在这个例子中，参数是 `out_features`，表示生成的张量的维度是 `out_features`。
                    2. 该函数生成的随机张量的元素值服从标准正态分布，即均值为 0，标准差为 1。
                    3. 生成的随机张量的元素值是在运行时动态生成的，每次运行结果可能会有所不同。
                torch.nn.Parameter 是 PyTorch 中用于定义可学习参数的类。通过将张量包装为 Parameter，可以将其纳入模型的参数管理中，以便在训练过程中进行优化。
        4. 如果 `bias` 为 `True`，则创建一个可学习的参数 `bias`，它是一个张量，形状为 `(out_features,)`，并使用正态分布进行初始化。
        5. 这些参数将在训练过程中通过反向传播进行更新，以实现对输入数据的线性变换。
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))

    # 将参数不需要训练的层在 forward 方法里(激活函数、Relu、+ - * /  ...)
    def forward(self, x):
        # mm-矩阵乘法  wx           将输入数据 x 与权重矩阵 self.weight 相乘, 实现了线性变换的核心计算。
        x = x.mm(self.weight)
        if self.bias:
            # wx + b              将偏置项添加到线性变换的结果中。expand_as(x) 用于将偏置项扩展为与输入数据 x 具有相同的形状。
            x = x + self.bias.expand_as(x)
        return x


'''
`if __name__ == '__main__':` 是 Python 中的一个特殊代码块，它的作用是在直接运行该脚本时执行特定的代码，而在被其他脚本导入时不执行。
    - `__name__` 是 Python 内置的一个变量，它表示当前模块的名称。当直接运行一个 Python 脚本时，`__name__` 的值被设置为 `__main__`。
                                                            当从其他脚本导入一个模块时，`__name__` 的值是被导入模块的名称。
    - `if __name__ == '__main__':` 后面的代码块只有在直接运行该脚本时才会执行。如果该脚本被其他脚本导入，那么这个代码块中的代码将不会被执行。
    这个代码块通常用于以下情况：
    1. 测试代码：在脚本中编写一些测试代码，只有在直接运行该脚本时才会执行这些测试代码。
    2. 主程序入口：如果脚本是一个可执行的程序，那么可以在这个代码块中编写主程序的逻辑。
    3. 模块的示例用法：在脚本中提供一些示例代码，展示如何使用该模块。这些示例代码只有在直接运行该脚本时才会执行。
    
    总之，`if __name__ == '__main__':` 提供了一种在直接运行脚本和被其他脚本导入时执行不同代码的机制。
'''
if __name__ == '__main__':
    # tarin for mnist
    net = Linear(3, 2)
    x = net.forward
    print('11', x)
