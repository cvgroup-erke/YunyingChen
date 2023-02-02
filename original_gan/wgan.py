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
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--gp", type=bool, default=True, help="use gradient penalty or not")
parser.add_argument("--save_dir", type=str, default='./resnet_gp', help="dir of result")
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)    

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Initialize generator and discriminator
# generator = Generator()
# discriminator = Discriminator()
generator = ResNet50_G()
discriminator = Discriminator()
if cuda:
    generator.cuda()
    discriminator.cuda()

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
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = imgs.type(Tensor)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        # z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], 3,7,7)))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        #gradient penalty 
        if opt.gp:
            alpha = torch.rand(real_imgs.shape[0],1,1,1)
            if cuda:
                alpha = alpha.cuda(non_blocking = True)
            x_hat = (alpha*real_imgs.data + (1-alpha)*fake_imgs).requires_grad_(True)
            out = discriminator(x_hat)
            dx = torch.autograd.grad(outputs=out,
                                     inputs=x_hat,
                                     grad_outputs=torch.ones(out.size()).cuda(),
                                     retain_graph = True,
                                     create_graph = True,
                                     only_inputs = True)[0].view(out.shape[0],-1)
            gp_loss = torch.mean((torch.norm(dx,p=2)-1)**2)
            loss_D  += gp_loss

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator while not using gradient penalty
        if not opt.gp:
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

        if batches_done % opt.sample_interval == 0:
            
            save_image(gen_imgs.data[:25], "%s/%d.png" % (opt.save_dir,batches_done), nrow=5, normalize=True)
        batches_done += 1

