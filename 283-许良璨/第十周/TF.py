import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

input_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,input_data.shape)
target=np.square(input_data)+noise


x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

L1W=tf.Variable(tf.random.normal([1,10]))
L1B=tf.Variable(tf.random.normal([1,10]))
L1_=tf.matmul(x,L1W)+L1B
L1=tf.nn.tanh(L1_)


L2W=tf.Variable(tf.random.normal([10,1]))
L2B=tf.Variable(tf.random.normal([1,1]))
L2_=tf.matmul(L1,L2W)+L2B
L2=tf.nn.tanh(L2_)



loss=tf.reduce_mean(tf.square(y-L2))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        sess.run(train_step,feed_dict={x:input_data,y:target})

    predection_value=sess.run(L2,feed_dict={x:input_data})


plt.figure()
plt.scatter(input_data,target)
plt.plot(input_data,predection_value,"g-",lw=5)
plt.show()
