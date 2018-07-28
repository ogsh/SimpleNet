import torch as th
import torch.nn as nn
from net.simplenet import NormalizationLayer

config_layer = [('c', 8),
                ('o', 0),
                ('m', 0),
                ('c', 16),
                ('o', 0),
                ('m', 0),
                ('c', 32),
                ('o', 0),
                ('m', 0),
                ('s', 64),
                ('o', 0)]


def conv2d(in_channels, out_channels, kernel_size):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)


def pool2():
    return nn.MaxPool2d(2, padding=0)


def resize_map(x, output_size):
    insize = [x.shape[2], x.shape[3]]

    scale = [output_size[0] / insize[0], output_size[1] / insize[1]]

    assert(scale[0] == scale[1])
    scale = scale[0]

    if scale > 1:
        outmap = nn.Upsample(x, scale_factor=scale)
    elif scale < 1:
        assert(insize[0] % output_size[0] == 0)
        scale2 = int(1 / scale)
        outmap = x[:, :, (scale2-1)::scale2, (scale2-1)::scale2]
    else:
        outmap = x

    return outmap


class OutputLayer(nn.Module):
    def __init__(self):
         super(OutputLayer, self).__init__()

    def forward(self, x):
        return x


class SparseConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, nsplit=2):
        super(SparseConv, self).__init__()

        assert(in_channels % nsplit == 0)
        assert(out_channels % nsplit == 0)

        self.sconv = nn.ModuleList()
        for i in range(nsplit):
            self.sconv += [conv2d(in_channels//nsplit, out_channels//nsplit, kernel_size=kernel_size)]

    def forward(self, x):
        x = th.chunk(x, 2, 1)

        y = [conv(x[i]) for i, conv in enumerate(self.sconv)]

        y = th.cat(y, 1)

        return y


class SimpleNet2(nn.Module):

    def __init__(self, is_train):
        super(SimpleNet2, self).__init__()
        self.is_train = is_train
        self.features = self.__make_layer(config_layer)
        self.classifier = nn.Sequential(
            self.__conv_bn_act(120, 3, 3),
            self.__conv_bn_act(3, 1, 1))
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

    def __conv_bn_act(self, in_channels, out_channels, kernel_size, has_activation=True, has_bias=False):
        module = nn.Sequential()

        padding = kernel_size // 2
        module.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=has_bias))

        module.add_module('bn', nn.BatchNorm2d(out_channels, affine=False))

        if has_activation:
            module.add_module('relu', nn.ReLU())

        return module

    def __make_layer(self, config):
        layers = nn.ModuleList()

        channels = 3
        for cfg in config:
            if cfg[0] == 'm':
                layers += [pool2()]
            elif cfg[0] == 'c':
                layers += [self.__conv_bn_act(channels, cfg[1], 3)]
                channels = cfg[1]
            elif cfg[0] == 's':
                layers += [SparseConv(channels, cfg[1], 3)]
                channels = cfg[1]
            elif cfg[0] == 'o':
                layers += [OutputLayer()]

        return layers

    def forward(self, x):
        output_size = [x.shape[2]//8, x.shape[3]//8]
        map_list = []

        net = x

        for idx, feature in enumerate(self.features):
            net = feature(net)
            if isinstance(feature, OutputLayer):
                map_list += [net]

        maps = [resize_map(map, output_size) for map in map_list]

        map = th.cat(maps, 1)

        map = self.classifier(map)


        return map



