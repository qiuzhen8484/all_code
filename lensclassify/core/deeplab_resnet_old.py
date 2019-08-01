import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
affine_par = True


def forwardhook(module, input, output):

    print 'Enter forward hook'
    print module


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
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

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
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


class Classifier_Module(nn.Module):

    def __init__(self,dilation_series,padding_series):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(2048,4,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, ):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__ = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24])

        #add hook
        #self.layer5.register_forward_hook(forwardhook)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                for i in m.parameters():
                    i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine = affine_par),
            )

        for i in downsample._modules['1'].parameters():
            i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))

        return nn.Sequential(*layers)

    def _make_pred_layer(self,block, dilation_series, padding_series):
        return block(dilation_series,padding_series)

    #get One Layer Feature map
    def GetfeatureMap(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    #fusionpart is a list [0,0,0,1,0] zero means no fusion, 1 means fusion
    #types if the fusion type
    def fusionforward(self, x, fusionpart, types):

        def fusionpart__(x, switchs, types):
            tensorsize = x.size()
            if switchs==0:
                return x
            #the fusion types:
            #1, sum average,
            #2, catannate dimention reduction,
            #3, max average
            if types == 1:
                x = torch.mean(x, 0)
                return x

            if types == 2:
                x = torch.chunk(tensor=x, chunks=tensorsize[0], dim=0)
                x = torch.cat(x, dim=1)
                return x

            if types == 3:
                x = torch.max(x, 0)
                return x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = fusionpart__(x, fusionpart[0],types)
        x = self.layer1(x)
        x = fusionpart__(x, fusionpart[1], types)
        x = self.layer2(x)
        x = fusionpart__(x, fusionpart[2], types)
        x = self.layer3(x)
        x = fusionpart__(x, fusionpart[3], types)
        x = self.layer4(x)
        x = fusionpart__(x, fusionpart[4], types)
        x = self.layer5(x)
        x = fusionpart__(x, fusionpart[5], types)
        return x


class MS_Deeplab(nn.Module):
    def __init__(self,block):
        super(MS_Deeplab,self).__init__()
        self.Model = ResNet(block,[3, 4, 23, 3])

    def forward(self,x):
        input_size = x.size()
        imwidth = input_size[2]
        imheight = input_size[3]

        self.interp1 = nn.UpsamplingBilinear2d(size = (int(imwidth*0.75),int(imheight*0.75))) #Scale 0.75
        self.interp2 = nn.UpsamplingBilinear2d(size = (int(imwidth*0.5), int(imheight*0.5)))  #Scale 0.5

        outOrigin = self.Model(x)
        # print input_size,outOrigin.size()

        self.interp3 = nn.UpsamplingBilinear2d(size=(imwidth,imheight))
        # self.interp3 = nn.UpsamplingBilinear2d(size=(outOrigin.size()[2], outOrigin.size()[3]))

        out075Origin = self.Model(self.interp1(x))
        out050Origin = self.Model(self.interp2(x))

        outOrigin = self.interp3(outOrigin)
        up075_FeatureMap = self.interp3(out075Origin)
        up050_FeatureMap = self.interp3(out050Origin)

        temp1 = torch.max(outOrigin,up075_FeatureMap)
        out4 = torch.max(temp1,up050_FeatureMap)

        # return outOrigin

        out=[]
        out.append(outOrigin)
        out.append(up075_FeatureMap)
        out.append(up050_FeatureMap)
        out.append(out4)

        return out


def Res_Deeplab():
    model = MS_Deeplab(Bottleneck)
    return model

