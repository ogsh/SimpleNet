import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable

class NormalizationLayer(nn.Module):
    def __init__(self, var, mean):
        super(NormalizationLayer, self).__init__()
        self.var = var
        self.mean = mean

    def forward(self, x):
        return (x - self.mean) / self.var

class SimpleNet(nn.Module):

    def __init__(self, is_train):
        super(SimpleNet, self).__init__()
        self.is_train = is_train

        self.feature1 = nn.ModuleList([
            self.__conv_bn_act(3, 8, 3),
            self.__conv_bn_act(8, 16, 3),
            self.__conv_bn_act(16, 32, 3),
            self.__conv_bn_act(32, 64, 3)
        ])

        self.sparse1 = nn.ModuleList([
            self.__conv_bn_act(32, 32, 3),
            self.__conv_bn_act(32, 32, 3)
        ])

        self.classifier = self.__conv_bn_act(64, 3, 3)

        self.__initiaize_weights()

    def __initiaize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, NormalizationLayer):
                nn.init.constant_(m.mean, 0.)
                nn.init.constant_(m.var, 0.5)

    def __pool2(self):
        return nn.MaxPool2d(2, padding=1)

    def __conv_bn_act(self, in_channels, out_channels, kernel_size, has_activation=True, has_bias=False):
        module = nn.Sequential()

        padding = kernel_size // 2
        module.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=has_bias))

        if self.is_train:
            module.add_module('bn', nn.BatchNorm2d(out_channels, affine=False))
        else:
            module.add_module('bn', NormalizationLayer(nn.Parameter(Tensor([1]), requires_grad=False), nn.Parameter(Tensor([1]), requires_grad=False)))

        if has_activation:
            module.add_module('relu', nn.ReLU())

        return module

    def forward(self, x):
        net = x

        feature_list = []
        for feature in self.feature1:
            net = feature(net)
            feature_list.append(net)
            net = self.__pool2()(net)

        net1, net2 = torch.chunk(net, 2, 1)
        net1 = self.sparse1[0](net1)
        net2 = self.sparse1[1](net2)
        net = torch.cat((net1, net2), 1)
        print(net.shape)

        net = self.__conv_bn_act(64, 3, 3)(net)

        return net




