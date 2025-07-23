# Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch.nn as nn
import math
from .custom_layers import MultiBatchNorm

from kan_convs import KANConv2DLayer


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    print("in conv3x3")
    return  KANConv2DLayer(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_types, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        print("in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = MultiBatchNorm('2d', bn_types, planes)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = MultiBatchNorm('2d', bn_types, planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_types, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        print("in Bottleneck")
        self.conv1 = KANConv2DLayer(inplanes, planes, kernel_size=1)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = MultiBatchNorm('2d', bn_types, planes)
        self.conv2 = KANConv2DLayer(planes, planes, kernel_size=3, stride=stride, padding=1)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = MultiBatchNorm('2d', bn_types, planes)
        self.conv3 = KANConv2DLayer(planes, planes * Bottleneck.expansion, kernel_size=1)
        # self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.bn3 = MultiBatchNorm('2d', bn_types, planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class KANResNetMultiBN(nn.Module):
    def __init__(self, dataset, depth, num_classes, bn_types, bottleneck=False):
        super(KANResNetMultiBN, self).__init__()
        self.dataset = dataset
        self.bn_types = bn_types

        if self.dataset.startswith('cifar'):
            print("in if cifar")
            self.inplanes = 16
            print(bottleneck)
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock

            print("n value: " + str(n))

            self.conv1 = KANConv2DLayer(3, self.inplanes, kernel_size=3, stride=1, padding=1)
            # self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.bn1 = MultiBatchNorm('2d', bn_types, self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, n)
            self.layer2 = self._make_layer(block, 32, n, stride=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2)
            # self.avgpool = nn.AvgPool2d(8)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64 * block.expansion, num_classes)

        elif dataset == 'imagenet':
            blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3],
                      200: [3, 24, 36, 3]}
            assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

            self.inplanes = 64

            self.conv1 = KANConv2DLayer(3, self.inplanes, kernel_size=7, stride=2, padding=3)
            # self.bn1 = nn.BatchNorm2d(64)
            self.bn1 = MultiBatchNorm('2d', bn_types, 64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
            self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
            self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
            self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
            # self.avgpool = nn.AvgPool2d(7)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                KANConv2DLayer(self.inplanes, planes * block.expansion,
                               kernel_size=1, stride=stride),
                # nn.BatchNorm2d(planes * block.expansion),
                MultiBatchNorm('2d', self.bn_types, planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.bn_types, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.bn_types))

        return nn.Sequential(*layers)

    def _set_bn_type(self, t):
        count = 0
        for m in self.modules():
            if isinstance(m, MultiBatchNorm):
                m.t = t
                count += 1

    def forward(self, x, t=None):

        if t is not None:
            self._set_bn_type(t)

        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x