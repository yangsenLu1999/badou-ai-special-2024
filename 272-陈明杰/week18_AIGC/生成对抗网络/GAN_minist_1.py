from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import matplotlib.pyplot as plt

import sys

import numpy as np




class GAN():
    def __init__(self, epochs=2000, batch_size=32, sample_interval=200):
        self.high = 28
        self.weight = 28
        self.channels = 1
        self.latent_dim = 100
        self.img_shape = (self.high, self.weight, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # 构建并compile辨别模型
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=optimizer,
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])

        # 构建并compile生成模型
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False
        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(optimizer=optimizer,
                              loss='binary_crossentropy')

        pass


    def build_generator(self):
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
        model.add(Reshape(target_shape=self.img_shape))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    # def build_generator(self):
    #
    #     model = Sequential()
    #
    #     model.add(Dense(256, input_dim=self.latent_dim))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(Dense(512))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(Dense(1024))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(Dense(np.prod(self.img_shape), activation='tanh'))
    #     model.add(Reshape(self.img_shape))
    #
    #     model.summary()
    #
    #     noise = Input(shape=(self.latent_dim,))
    #     img = model(noise)
    #
    #     return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        # 这里的sigmoid不能用relu，否则训练不收敛
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=(self.high, self.weight, self.channels))
        validity = model(img)
        return Model(img, validity)

        pass

    def train(self, epochs, batch_size=128, sample_interval=50):
        # 加载数据集
        (X_train, _), (_, _) = mnist.load_data()
        # 标准化
        X_train = X_train / 127.5 - 1.0
        # 添加维度
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # 生成随机下标
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, size=(batch_size, self.latent_dim))
            z = self.generator.predict(noise)

            loss1 = self.discriminator.train_on_batch(imgs, valid)
            loss2 = self.discriminator.train_on_batch(z, fake)
            d_loss = 0.5 * np.add(loss1, loss2)

            noise = np.random.normal(0, 1, size=(batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            # print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, loss[0], 100 * loss[1], g_loss))
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            # if epoch % sample_interval == 0:
            #     self.sample_images(epoch)
            pass
        pass




if __name__ == '__main__':
    gen = GAN()
    gen.train(epochs=2000, batch_size=32, sample_interval=200)
