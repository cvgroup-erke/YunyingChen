# dataset: mnist
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch

from generator import Generator
from discriminator import Discriminator
from resnet50 import ResNet50_G,ResNet50_D
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Loss function
adversarial_loss = torch.nn.BCELoss()       # binary cross entropy loss: 2分类

# Initialize generator and discriminator
generator = ResNet50_G()
discriminator = ResNet50_D()
print(discriminator)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]    # [] means channel, 0.5,0.5 means mean & std
                                                    # => img = (img - mean) / 0.5 per channel
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):      # batch id, (image, target)

        # Adversarial ground truths
        # Variable aims to wrap tensor and provides the auto-bp function.
        # It's now deprecated
        # valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False
        #                                          fill self tensor with specific value
        # fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        # tensor: [batch, channel, width, height]
        valid = Tensor(imgs.size(0), 1).fill_(1.0).detach()     # detach means "requires_grad = False"
        fake = Tensor(imgs.size(0), 1).fill_(0.0).detach()

        # Configure input
        # real_imgs = Variable(imgs.type(Tensor))
        real_imgs = imgs.type(Tensor)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()     # 对已有的gradient清零(因为来了新的batch_size的image)

        # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], 3,7,7)))

        # Generate a batch of images
        gen_imgs = generator(z)     # G(z) ——> D(G(z)), gen_imgs ~ pg

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs),  # D(G(z))
                                  valid)                    # label = 1

        g_loss.backward()       # bp, 算gradient， x.grad += dloss/dx
        optimizer_G.step()      # 更新x， x -= lr * x.grad

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs),  # D(x)
                                     valid)                     # label = 1
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()),  # D(G(z))
                                     fake)                              # label = 0
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images_output/%d.png" % batches_done, nrow=5, normalize=True)
