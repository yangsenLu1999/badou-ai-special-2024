import tensorflow as tf
# 了解tensorflow基础的一些概念，用法

# TensorFlow Python 库有一个默认图 (default graph), op 构造器可以为其增加节点.
# 这个默认图对许多程序来说已经足够用了.
# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点加到默认图中.
#

# ================= 常量张量 =============================

# 构造器的返回值代表该常量张量 op 的返回值. 1x2 矩阵
matrix1 = tf.constant([[3., 3.]])

# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2.], [2.]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.
product = tf.matmul(matrix1, matrix2)

'''
上述过程只是定义了张量和操作，不会立即执行计算，只是建立了计算图。

默认图现在有三个节点, 两个 constant() op, 和一个matmul() op. 
为了真正进行矩阵相乘运算, 并得到矩阵乘法的结果, 必须在【会话里启动】这个图.
启动图的第一步是创建一个 Session 对象, 如果无任何创建参数, 会话构造器将启动默认图.
'''
# 启动默认图. tensorflow1.x需要使用session的方式计算。2.x版本不需要
sess = tf.Session()

# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数.
# 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回矩阵乘法 op 的输出.
#
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
#
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
#
# 返回值 'result' 是一个 numpy `ndarray` 对象.
result = sess.run(product)
print(sess.run(matrix2)) # 也可以通过这种方式打印值
print(result)
# ==> [[ 12.]]

# 任务完成, 关闭会话.
sess.close()
'''
session对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 “with” 代码块来自动完成关闭动作.

使用上下文管理器
with tf.Session() as sess:
  result = sess.run([product])
  print (result)
'''


# ================= 变量 =============================

# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1
one = tf.constant(1)
new_value = tf.add(state, one)

# 将 state 的值更新为 new_value，tf.assign 函数会为指定的变量分配一个新值
# tf.compat.v1.assign
update = tf.assign(state, new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
# tf.compat.v1.global_variables_initializer()
init_op = tf.global_variables_initializer()

# 启动图, 运行 op
with tf.Session() as sess:
    # 运行 'init' op
    sess.run(init_op)
    # 打印 'state' 的初始值
    print("state", sess.run(state))
    # 运行 op, 更新 'state', 并打印 'state'
    for _ in range(5):
        sess.run(update)
        print("update:", sess.run(state)) # 实现的是一个计数器的功能


'''Tips
在 TensorFlow 中，特别是在 TensorFlow 1.x 中，操作并不是立即执行的，而是构建一个计算图（graph）。
每次的sess.run(update)运行，都是执行了一系列的操作(本例中：加法+赋值)，而不是简单的单行赋值。是需要通过整个图的逻辑计算，才得到新值的。

'''


# ================= TensorBoard =============================

# 定义一个计算图，实现两个向量的减法操作
# 定义两个输入，a为常量，b为变量
'''
a = tf.constant([10.0, 20.0, 40.0], name='a')
b = tf.Variable(tf.random_uniform([3]), name='b')
output = tf.add_n([a, b], name='add')
'''


# 生成一个具有写权限的日志文件操作对象，将当前命名空间的计算图写进日志中
'''
writer = tf.summary.FileWriter('logs', tf.get_default_graph())
writer.close()
'''

# 启动tensorboard服务（在命令行启动）。
# tensorboard --logdir logs

# 启动tensorboard服务后，复制地址并在本地浏览器中打开，可以将流程可视化出来


# =================== fetch =============================

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed) # 乘法运算

with tf.Session() as sess:
    result = sess.run([mul, intermed])  # 需要获取的多个 tensor 值，在 op 的一次运行中一起获得（而不是逐个去获取 tensor）。
    print(result)


# =================== feed =============================

# tf.compat.v1.placeholder
input1 = tf.placeholder(tf.float32) # 占位符（placeholder）可以实现动态输入，后续通过feed_dict传入实际数据
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2) # 乘法

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1: [7], input2: [2.]})) # 喂数据

# 输出:
# [array([ 14.], dtype=float32)]