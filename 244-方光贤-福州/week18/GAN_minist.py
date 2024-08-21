from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GAN():
    def __init__(self):
        # 定义图片属性 还定义了潜在空间的维度用于生成器生成图片的随机噪声向量
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # 定义Adam优化器的参数
        optimizer = Adam(0.0002, 0.5)

        # 构建判别器 设置判别器的损失函数为二元交叉熵因为是二分类任务 指标为accuracy
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # 构建生成器
        self.generator = self.build_generator()

        # 以噪声作为输入学习生成以假乱真的图片
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # 首先只训练生成器 我们要固定判别器
        self.discriminator.trainable = False

        # 判别器以生成器的输出作为输入来验证图片
        validity = self.discriminator(img)

        # 创建一个组合模型，用于同时训练生成器和判别器
        # 输入是噪声 输出是判别器对生成图片的判断
        # 不关心accuracy 因为主要关心生成器和判别器的博弈 并不关心图片质量
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):
        # 定义Sequential模型 使用Dense层 激活函数为LeakyRelu 归一化 输入为噪声 最后reshape成图片形状
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        # 定义Sequential模型 先拉直再连接Dense 使用LeakyRelu激活 用sigmoid输出判别结果
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    # 定义训练函数 参数包括轮次 批次 样本周期
    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # 归一化添加通道维度
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # 设置真假
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # 随机挑选图片
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            # 随机生成噪声
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # 生成假图像
            gen_imgs = self.generator.predict(noise)

            # 训练判别器 真实的图片标签就给1 虚假的图片标签就给0 计算损失各50%
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 再随机生成一批新噪声
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 训练联合模型
            g_loss = self.combined.train_on_batch(noise, valid)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # 每个样本周期保存一下
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        # 生成随机噪声 并显示图像
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    # 训练GAN
    gan = GAN()
    gan.train(epochs=2000, batch_size=32, sample_interval=200)
