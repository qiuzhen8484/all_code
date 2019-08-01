from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
import math
from torch.nn import init



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*53*53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*53*53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features


def train(model, loss, optimizer, x_val, y_val):
    model.train()
    x = Variable(x_val.cuda())
    y = Variable(y_val.cuda())

    optimizer.zero_grad()
    output = model.forward(x)
    hloss = loss.forward(output, y)
    hloss.backward()
    optimizer.step()
    #print(output.size())
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == y).sum()
    # distance_loss = []
    # for i in range(len(x_val)):
    #     distance_loss += [abs(predicted[i]-y[i])]

    return hloss.data[0], correct

def predict(model,x_val):
    x=Variable(x_val.cuda())
    output=model.forward(x)
    return output.gpu().data.numpy().argmax(axis=1)

def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x

class ShuffleNetUnitA(nn.Module):
    """ShuffleNet unit for stride=1"""
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitA, self).__init__()
        assert in_channels == out_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                        1, groups=groups, stride=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=1,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        out = F.relu(x + out)
        return out

class ShuffleNetUnitB(nn.Module):
    """ShuffleNet unit for stride=2"""
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitB, self).__init__()
        out_channels -= in_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                     1, groups=groups, stride=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=2,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        out = F.relu(torch.cat([x, out], dim=1))
        return out

class ShuffleNet(nn.Module):
    """ShuffleNet for groups=3"""
    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        super(ShuffleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 24, 3, stride=2, padding=1)
        stage2_seq = [ShuffleNetUnitB(24, 240, groups=3)] + \
            [ShuffleNetUnitA(240, 240, groups=3) for i in range(3)]
        self.stage2 = nn.Sequential(*stage2_seq)
        stage3_seq = [ShuffleNetUnitB(240, 480, groups=3)] + \
            [ShuffleNetUnitA(480, 480, groups=3) for i in range(7)]
        self.stage3 = nn.Sequential(*stage3_seq)
        stage4_seq = [ShuffleNetUnitB(480, 960, groups=3)] + \
                     [ShuffleNetUnitA(960, 960, groups=3) for i in range(3)]
        self.stage4 = nn.Sequential(*stage4_seq)
        # self.fc1 = nn.Linear(960, num_classes)
        self.fc1 = nn.Linear(2880, num_classes)

    def forward(self, x):
        net = self.conv1(x)
        net = F.max_pool2d(net, 3, stride=2, padding=1)
        net = self.stage2(net)
        net = self.stage3(net)
        net = self.stage4(net)
        # net = F.avg_pool2d(net, 7)
        net = F.avg_pool2d(net, 6)
        net = net.view(net.size(0), -1)
        net = self.fc1(net)
        logits = net
        return logits


class ShuffleNet_avgpool(nn.Module):
    """ShuffleNet for groups=3"""
    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        super(ShuffleNet_avgpool, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 24, 3, stride=2, padding=1)
        stage2_seq = [ShuffleNetUnitB(24, 240, groups=3)] + \
            [ShuffleNetUnitA(240, 240, groups=3) for i in range(3)]
        self.stage2 = nn.Sequential(*stage2_seq)
        stage3_seq = [ShuffleNetUnitB(240, 480, groups=3)] + \
            [ShuffleNetUnitA(480, 480, groups=3) for i in range(7)]
        self.stage3 = nn.Sequential(*stage3_seq)
        stage4_seq = [ShuffleNetUnitB(480, 960, groups=3)] + \
                     [ShuffleNetUnitA(960, 960, groups=3) for i in range(3)]
        self.stage4 = nn.Sequential(*stage4_seq)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(984, num_classes)

    def forward(self, x):
        net = self.conv1(x)
        net1 = net
        net1 = self.avgpool(net1)
        net1 = net1.view(net1.size(0), -1)
        net = F.max_pool2d(net, 3, stride=2, padding=1)
        net = self.stage2(net)
        net = self.stage3(net)
        net = self.stage4(net)
        net = self.avgpool(net)
        net = net.view(net.size(0), -1)
        net = torch.cat([net, net1], 1)
        net = self.fc1(net)
        logits = net
        return logits

class Bottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(Bottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)
        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D*C)
        self.conv3 = nn.Conv2d(D*C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

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


class ResNeXt(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, baseWidth, cardinality, layers, num_classes):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(ResNeXt, self).__init__()
        block = Bottleneck

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext50(baseWidth, cardinality):
    """
    Construct ResNeXt-50.
    """
    model = ResNeXt(baseWidth, cardinality, [3, 4, 6, 3], 2)
    return model