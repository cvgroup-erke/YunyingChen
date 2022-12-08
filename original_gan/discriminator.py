import torch.nn as nn
import numpy as np

img_shape = (1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img): # img.shape = torch.Size([64, 1, 28, 28]) = 64 * 1 * 28 * 28
        img_flat = img.view(img.size(0), -1)    # (64, -1 = 1 * 28 * 28)
        validity = self.model(img_flat)
        return validity
