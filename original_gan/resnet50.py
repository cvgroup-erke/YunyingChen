import torchvision
import torch.nn as nn
import numpy as np
import torch

import torchvision.models._utils as _utils
from torch.hub import load_state_dict_from_url


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}

img_shape = (1, 28, 28)


class ResNet50_D(nn.Module):
    def __init__(self,pretrain= True):
        super(ResNet50_D,self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=pretrain)
        self.backbone = _utils.IntermediateLayerGetter(pretrained,{'layer1':1})
        self.conv1x1 = nn.Conv2d(256, 1024, kernel_size=7, stride=1, bias=False)
        self.fc1 = nn.Linear(1024, 512)
        self.LeakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(512,1)
        self.sigmoid  = nn.Sigmoid()
    
    def forward(self,input):
        img_rgb = torch.stack((input,)*3,axis=2)
        img_rgb = torch.squeeze(img_rgb)
        x = self.backbone(img_rgb)
        x = list(x.values())
        x = self.conv1x1(x[0])
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.LeakyRelu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class ResNet50_G(nn.Module):
    output_size = int(np.prod(img_shape))

    def __init__(self, pretrain= True):
        super(ResNet50_G,self).__init__()
        pretrained = torchvision.models.resnet50()
        if pretrain:
            pretrained_dict = load_state_dict_from_url(model_urls['resnet50'],progress=True)
            # pretrained_dict = torch.load('./resnet50-19c8e357.pth')
            pretrained.load_state_dict(pretrained_dict)
        self.backbone = _utils.IntermediateLayerGetter(pretrained,{'layer4': 1})

        self.conv1x1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False)
        self.fc = nn.Linear(1024, int(np.prod(img_shape)))
        self.tanh = nn.Tanh()
        
    def forward(self,input):
        input = self.backbone(input)
        x = list(input.values())
        x = self.conv1x1(x[0])
        x = torch.squeeze(x)
        x = self.fc(x)
        x = self.tanh(x)

        img = x.view(x.size(0), *img_shape)
        return img
