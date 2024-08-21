import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class Config:
    img_shape = (1, 28, 28)
    latent_dim = 100
    lr = 0.0002
    batch_size = 64
    epochs = 50
    sample_interval = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define the generator
class Generator(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.img_shape = config.img_shape

        self.gen = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256, 0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512, 0.8),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024, 0.8),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.gen(x)
        return x.view(x.size(0), *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.img_shape = config.img_shape

        self.disc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.img_shape), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

# define the GAN model
class GAN():
    def __init__(self, config:Config, train_data, val_data):
        self.config = config

        self.generator = Generator(config).to(config.device)
        self.discriminator = Discriminator(config).to(config.device)
        
        self.gen_optim = torch.optim.Adam(self.generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.discri_optim = torch.optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.loss_fn = nn.BCELoss()

        self.train_data = train_data
        self.val_data = val_data

    def train_discriminator(self, real_imgs, valid_label, fake_label, fake_imgs):
        self.discri_optim.zero_grad()

        real_loss = self.loss_fn(self.discriminator(real_imgs), valid_label)
        fake_loss = self.loss_fn(self.discriminator(fake_imgs.detach()), fake_label)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        self.discri_optim.step()
        return d_loss.item()

    def train_generator(self, valid_label, discri_label):
        self.gen_optim.zero_grad()
        g_loss = self.loss_fn(discri_label, valid_label)
        g_loss.backward()
        self.gen_optim.step()
        return g_loss.item()
    
    @torch.no_grad()
    def sample(self, epoch):
        self.generator.eval()
        
        r, c = 5, 5
        noise = torch.randn(r*c, self.config.latent_dim).to(self.config.device)
        gen_imgs = self.generator(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt=0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt].detach().cpu().numpy().reshape(28, 28), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f'images/{epoch}.png')
        plt.close()


    def train(self):
        pbar = tqdm(total=self.config.epochs)
        valid_label = torch.ones(self.config.batch_size, 1).to(self.config.device)
        fake_label = torch.zeros(self.config.batch_size, 1).to(self.config.device)
        for epoch in range(self.config.epochs):
            for _, (imgs, _) in enumerate(self.train_data):
                real_imgs = imgs.to(self.config.device)
                noise = torch.randn(self.config.batch_size, self.config.latent_dim).to(self.config.device)
                fake_imgs = self.generator(noise)

                d_loss = self.train_discriminator(real_imgs, valid_label, fake_label, fake_imgs)

                discri_label = self.discriminator(fake_imgs)
                g_loss = self.train_generator(valid_label, discri_label)

            if epoch % self.config.sample_interval == 0:
                self.sample(epoch)
            
            pbar.set_description(f'Epoch: {epoch}, D_loss: {d_loss}, G_loss: {g_loss}')
            pbar.update(1)
        pbar.close()

if __name__ == '__main__':
    config = Config()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    val_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    
    train_loder = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, drop_last=True)

    gan = GAN(config, train_loder, val_loader)
    gan.train()


