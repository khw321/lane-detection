import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from unet import unet_parts
import torch
import time
__all__ = [
    'VGG', 'vgg16_bn'
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(0.1)
        self.scnn = unet_parts.scnn(128, 128)
        self.out = nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # n, 512, 80, 60
        x = self.conv5(x)  # n, 512, 80, 60
        x = self.fc1(x)  # n, 128, 80, 60
        x = self.scnn(x)
        x = self.drop(x)
        x = self.out(x)  # n, 2, 80, 60
        x = self.up(x)
        x = self.up(x)
        x = self.up(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG_gn(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG_gn, self).__init__()
        self.features = features
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            GroupBatchnorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            GroupBatchnorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            GroupBatchnorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=4, dilation=4),
            GroupBatchnorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1),
            GroupBatchnorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.scnn = unet_parts.scnn(128, 128)
        self.drop = nn.Dropout2d(0.1)
        self.out = nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # n, 512, 80, 60
        x = self.conv5(x)  # n, 512, 80, 60
        x = self.fc1(x)  # n, 128, 80, 60
        #x = self.scnn(x)
        x = self.drop(x)
        x = self.out(x)  # n, 2, 80, 60
        x = self.up(x)
        x = self.up(x)
        x = self.up(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG_freespace_gn(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG_freespace_gn, self).__init__()
        self.features = features
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            GroupBatchnorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            GroupBatchnorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            GroupBatchnorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=4, dilation=4),
            GroupBatchnorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1),
            GroupBatchnorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.lane = self.lane_model()
        self.freespace = self.freespace_model()
        self.scnn = unet_parts.scnn(128, 128)
        self.drop = nn.Dropout2d(0.1)
        self.out = nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # n, 512, 80, 60
        x = self.conv5(x)  # n, 512, 80, 60
        # x = self.fc1(x)  # n, 128, 80, 60
        # x = self.scnn(x)
        # x = self.drop(x)
        # x = self.out(x)  # n, 2, 80, 60
        lane = x
        freespace = x
        lane_out = self.lane(lane)
        freespace_out = self.freespace(freespace)
        x = torch.cat((lane_out, freespace_out), dim=1)
        x = self.up(x)
        x = self.up(x)
        x = self.up(x)
        return x

    def lane_model(self):
        fc1 = [nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=4, dilation=4),
            GroupBatchnorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1),
            GroupBatchnorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0)
        ]
        return nn.Sequential(*fc1)

    def freespace_model(self):
        fc1 = [nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=4, dilation=4),
            GroupBatchnorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1),
            GroupBatchnorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0)
        ]
        return nn.Sequential(*fc1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num, group_num=32, eps=1e-10):
        super(GroupBatchnorm2d, self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, self.group_num, -1)

        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)

        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta


def make_layers(cfg, batch_norm=False, group_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            elif group_norm:
                layers += [conv2d, GroupBatchnorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model

def vgg16_gn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_gn(make_layers(cfg['D'], group_norm=True), **kwargs)
    return model

def vgg16_freespace_gn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_freespace_gn(make_layers(cfg['D'], group_norm=True), **kwargs)
    return model
