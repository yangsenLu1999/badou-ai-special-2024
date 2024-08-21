import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten, LeakyReLU, Input, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.models import Model
from keras.datasets import mnist
from matplotlib import pyplot as plt


class GAN():
    def __init__(self):
        self.img_row = 28
        self.img_col = 28
        self.channels = 1
        self.img_shape = (self.img_row, self.img_col, self.channels)
        self.latent_dim = 100
        self.pixel_num = self.img_row * self.img_col * self.channels

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim, ))
        x = self.generator(z)
        self.discriminator.trainable = False
        validity = self.discriminator(x)

        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1024))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.pixel_num, activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.latent_dim, ))
        img = model(noise)

        return Model(noise, img)

    def train(self, epochs=2000, batch_size=128, sample_interval=100):
        (train_data, _), (_, _) = mnist.load_data()

        train_data = train_data / 127.5 - 1
        train_data = np.expand_dims(train_data, axis=3)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            index = np.random.randint(0, train_data.shape[0], batch_size)
            images = train_data[index]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_images = self.generator.predict(noise)
            loss_real = self.discriminator.train_on_batch(images, valid)
            loss_fake = self.discriminator.train_on_batch(fake_images, fake)
            loss_d = 0.5 * np.add(loss_real, loss_fake)

            g_loss = self.combined.train_on_batch(noise, valid)
            print(f'{epoch} discriminator loss: {loss_d[0]:.4f}, generator loss: {g_loss:.4f}')

            if epoch % sample_interval == 0:
                self.sample_image(epoch)

    def sample_image(self, epoch):
        row, col = 5, 5
        noise = np.random.normal(0, 1, (row * col, self.latent_dim))
        image = self.generator.predict(noise)
        fig, ax = plt.subplots(row, col)
        count = 0
        for i in range(row):
            for j in range(col):
                ax[i, j].imshow(image[count, :, :, :], cmap='gray')
                ax[i, j].axis('off')
                count += 1
        fig.savefig(f'images/sample_{epoch}.png')
        plt.close()

if __name__ == '__main__':
    model = GAN()
    model.train(epochs=2000, batch_size=32, sample_interval=200)
