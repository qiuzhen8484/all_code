import os

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

from blocks import StackEncoder
from blocks import StackDecoder
from blocks import ConvBnRelu2d
from blocks import ResStackDecoder
from blocks import M_Encoder
from blocks import M_Decoder
from blocks import Decoder
from blocks import M_Conv
from blocks import MobileNetEncoder, MobileNetDecoder, MobileNet_block

from blocks import M_Decoder_my,M_Decoder_my_2,M_Decoder_my_3,M_Encoder_my_1,M_Decoder_my_4,M_Decoder_my_5,M_Decoder_my_6,M_Decoder_my_7,M_Decoder_my_8,M_Decoder_my_9,M_Decoder_my_10

from resNet_model import resnet50
from VGG_model import vgg16

from guided_filter_pytorch.guided_filter import GuidedFilter, FastGuidedFilter
from guided_filter_pytorch.guided_filter_attention import FastGuidedFilter_attention
from guided_filter_pytorch.guided_filter_my import FastGuidedFilter_my
from guided_filter_pytorch.box_filter import BoxFilter
import numpy as np
import math

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# ------------------------------------U-Net--------------------------------------------------------------------
# baseline 128x128, 256x256, 512x512, 1024x1024 for experiments -----------------------------------------------

class M_Net(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up(nn.Module):
    """
    compare Guided Filter with skip-connection and upsample
    author: Shihao Zhang
    Data: 2019/1/13
    """
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my(64 , 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv4, up5,conv3)
        up7 = self.up7(conv3, up6,conv2)
        up8 = self.up8(conv2, up7,conv1)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up_2(nn.Module):
    """
    compare Guided Filter with skip-connection and upsample
    author: Shihao Zhang
    Data: 2019/1/14
    """
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_2, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_2(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_2(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_2(64 , 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv4, up5,conv3)
        up7 = self.up7(conv3, up6,conv2)
        up8 = self.up8(conv2, up7,conv1)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up_3(nn.Module):
    """
    compare Guided Filter with skip-connection and upsample
    author: Shihao Zhang
    Data: 2019/1/14
    """
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_3, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_3(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_3(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_3(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_3(64 , 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out1 = self.down1(x)
        out = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out)
        out = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out)
        out = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out)
        out = self.center(out4)
        up5 = self.up5(out4, out, conv4)
        up6 = self.up6(out3, up5,conv3)
        up7 = self.up7(out2, up6,conv2)
        up8 = self.up8(out1, up7,conv1)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up_4(nn.Module):
    """
    compare Guided Filter with skip-connection and upsample
    author: Shihao Zhang
    Data: 2019/1/16
    """
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_4, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder_my_1(128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder_my_1(256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out1 = self.down1(x)

        x_2 = self.conv2(x_2)
        out = torch.cat([x_2, out1], dim=1)
        conv2, out2 = self.down2(out)

        x_3 = self.conv3(x_3)
        conv3, out3 = self.down3(x_2,x_3,conv2)

        x_4 = self.conv4(x_4)
        conv4, out4 = self.down4(x_3,x_4,conv3)

        out = self.center(out4)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up_5(nn.Module):
    """
    compare Guided Filter with skip-connection and upsample
    author: Shihao Zhang
    Data: 2019/1/16
    """
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_5, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_4(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_4(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_2(64 , 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)

        x_2 = self.conv2(x_2)
        out = torch.cat([x_2, out], dim=1)
        conv2, out = self.down2(out)

        x_3 = self.conv3(x_3)
        out = torch.cat([x_3, out], dim=1)
        conv3, out = self.down3(out)

        x_4 = self.conv4(x_4)
        out = torch.cat([x_4, out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv4, up5,conv3,x_3)
        up7 = self.up7(conv3, up6,conv2,x_2)
        up8 = self.up8(conv2, up7,conv1)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up_5_AT(nn.Module):
    """
    compare Guided Filter with Attention Guided Filter
    author: Shihao Zhang
    Data: 2019/1/16
    """
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_5_AT, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64 , 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter_attention(r=2,eps=0.01)

        # attention blocks
        self.attentionblock6 = GridAttentionBlock(in_channels=256)
        self.attentionblock7 = GridAttentionBlock(in_channels=128)
        self.attentionblock8 = GridAttentionBlock(in_channels=64)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)

        x_2 = self.conv2(x_2)
        out = torch.cat([x_2, out], dim=1)
        conv2, out = self.down2(out)

        x_3 = self.conv3(x_3)
        out = torch.cat([x_3, out], dim=1)
        conv3, out = self.down3(out)

        x_4 = self.conv4(x_4)
        out = torch.cat([x_4, out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up5 = self.gf(conv4, up5, torch.cat([x_3, conv3], dim=1),self.attentionblock6(conv4,up5))
        up6 = self.up6(up5)
        up6 = self.gf(conv3, up6, torch.cat([x_2, conv2], dim=1),self.attentionblock7(conv3,up6))
        up7 = self.up7(up6)
        up7 = self.gf(conv2, up7, torch.cat([conv1, conv1], dim=1),self.attentionblock8(conv2,up7))
        up8 = self.up8(up7)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up_6(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_6, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_5(512 , 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_5(256 , 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_5(128 , 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)

        x_2 = self.conv2(x_2)
        out = torch.cat([x_2, out], dim=1)
        conv2, out = self.down2(out)

        x_3 = self.conv3(x_3)
        out = torch.cat([x_3, out], dim=1)
        conv3, out = self.down3(out)

        x_4 = self.conv4(x_4)
        out = torch.cat([x_4, out], dim=1)
        conv4, out = self.down4(out)

        out = self.center(out)
        up5 = self.up5(conv4, out, x_4)
        up6 = self.up6(conv3, up5, x_3)
        up7 = self.up7(conv2, up6, x_2)
        up8 = self.up8(conv1, up7)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)


        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up_7(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_7, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.gf = FastGuidedFilter(r=2, eps=0.8)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)

        x_2 = self.conv2(x_2)
        out = torch.cat([x_2, out], dim=1)
        conv2, out = self.down2(out)

        x_3 = self.conv3(x_3)
        out = torch.cat([x_3, out], dim=1)
        conv3, out = self.down3(out)

        x_4 = self.conv4(x_4)
        out = torch.cat([x_4, out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)

        up5 = self.up5(conv4, out)
        up5 = self.gf(x_4,up5,x_4)

        up6 = self.up6(conv3, up5)
        up6 = self.gf(x_3,up6,x_3)

        up7 = self.up7(conv2, up6)
        up7 = self.gf(x_2,up7,x_2)

        up8 = self.up8(conv1, up7)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]

class M_Net_Up_8(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_8, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.gf = FastGuidedFilter(r=2,eps=0.01)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up5 = self.gf(conv4, up5, conv4)

        up6 = self.up6(conv3, up5)
        up6 = self.gf(conv3, up6, conv3)

        up7 = self.up7(conv2, up6)
        up7 = self.gf(conv2, up7, conv2)

        up8 = self.up8(conv1, up7)
        up8 = self.gf(conv1, up8, conv1)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up_8_AT(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_8_AT, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # self.AT1 = PAM_Module(32)
        # self.AT2 = PAM_Module(64)
        # self.AT3 = PAM_Module(128)
        # self.AT4 = PAM_Module(256)
        self.gating = nn.Sequential(nn.Conv2d(512, 256, 1),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       )
        # attention blocks
        self.attentionblock2 = GridAttentionBlock(in_channels=filters[1], gating_channels=filters[3],
                                                    inter_channels=filters[1], sub_sample_factor=attention_dsample, mode=nonlocal_mode)
        self.attentionblock3 = GridAttentionBlock(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2], sub_sample_factor=attention_dsample, mode=nonlocal_mode)
        self.attentionblock4 = GridAttentionBlock(in_channels=filters[3], gating_channels=filters[3],
                                                    inter_channels=filters[3], sub_sample_factor=attention_dsample, mode=nonlocal_mode)

        self.AT1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, bias=True)
        self.AT2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=True)
        self.AT3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=True)
        self.AT4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=True)
        self.gf = FastGuidedFilter_attention(r=2,eps=0.01)
        self.softmax = nn.Softmax(dim=-1)
        self.Relu = nn.ReLU()

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)

        gating = self.gating(out)




        up5 = self.up5(conv4, out)
        up5 = self.gf(conv4, up5, conv4, self.Relu(self.AT4(conv4)) )

        up6 = self.up6(conv3, up5)
        up6 = self.gf(conv3, up6, conv3, self.Relu(self.AT3(conv3)) )

        up7 = self.up7(conv2, up6)
        up7 = self.gf(conv2, up7, conv2, self.Relu(self.AT2(conv2)) )

        up8 = self.up8(conv1, up7)
        up8 = self.gf(conv1, up8, conv1, self.Relu(self.AT1(conv1)) )

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up_9(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_9, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_8(256 , 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_8(128 , 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_8(64 , 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.gf = FastGuidedFilter(r=2,eps=0.01)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv4, up5)
        up7 = self.up7(conv3, up6)
        up8 = self.up8(conv2, up7)
        up8 = self.gf(conv1,up8,conv1)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up_10(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_10, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_8(256 , 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_8(128 , 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_8(64 , 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.gf = FastGuidedFilter(r=2,eps=0.01)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv4, up5)
        up7 = self.up7(conv3, up6)
        up8 = self.up8(conv2, up7)
        up8 = self.gf(conv1.up8,conv1)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up_11(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_11, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_9(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_9(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_9(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_9(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.gf = FastGuidedFilter(r=2,eps=0.01)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(out)
        # up5 = self.gf(self.conv4(x_4), up5, self.conv4(x_4))
        up5 = self.gf(up5, self.conv4(x_4), up5)

        up6 = self.up6(up5)
        # up6 = self.gf(self.conv3(x_3), up6, self.conv3(x_3))
        up6 = self.gf(up6, self.conv3(x_3), up6)

        up7 = self.up7(up6)
        # up7 = self.gf(self.conv2(x_2), up7, self.conv2(x_2))
        up7 = self.gf(up7, self.conv2(x_2), up7)

        up8 = self.up8(up7)
        up8 = self.gf(conv1, up8, conv1)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up_12(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_12, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter(r=2,eps=0.01)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)

        feature_guidance = torch.cat([self.conv4(x_4), conv4], dim=1)
        temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        out = F.upsample(out, size=(temp_H, temp_W), mode='bilinear')
        out = self.gf(feature_guidance, out, feature_guidance)
        up5 = self.up5(out)

        feature_guidance = torch.cat([self.conv3(x_3), conv3], dim=1)
        temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        out = F.upsample(up5, size=(temp_H, temp_W), mode='bilinear')
        out = self.gf(feature_guidance, out, feature_guidance)
        up6 = self.up6(out)

        feature_guidance = torch.cat([self.conv2(x_2), conv2], dim=1)
        temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        out = F.upsample(up6, size=(temp_H, temp_W), mode='bilinear')
        out = self.gf(feature_guidance, out, feature_guidance)
        up7 = self.up7(out)

        feature_guidance = torch.cat([conv1, conv1], dim=1)
        temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        out = F.upsample(up7, size=(temp_H, temp_W), mode='bilinear')
        out = self.gf(feature_guidance, out, feature_guidance)
        up8 = self.up8(out)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Up_12_AT(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Up_12_AT, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter_attention(r=2,eps=0.01)

        # attention blocks
        self.attentionblock5 = GridAttentionBlock(in_channels=512)
        self.attentionblock6 = GridAttentionBlock(in_channels=256)
        self.attentionblock7 = GridAttentionBlock(in_channels=128)
        self.attentionblock8 = GridAttentionBlock(in_channels=64)

        # self.attentionblock5 = GridAttentionBlock_2(in_channels=512,gating_channels=512)
        # self.attentionblock6 = GridAttentionBlock_2(in_channels=256,gating_channels=512)
        # self.attentionblock7 = GridAttentionBlock_2(in_channels=128,gating_channels=512)
        # self.attentionblock8 = GridAttentionBlock_2(in_channels=64,gating_channels=512)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        out_gate = out

        # feature_guidance = torch.cat([self.conv4(x_4), conv4], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # out = F.upsample(out, size=(temp_H, temp_W), mode='bilinear')
        # out = self.gf(feature_guidance, out, feature_guidance,self.attentionblock5(feature_guidance,out))
        # up5 = self.up5(out)
        #
        # feature_guidance = torch.cat([self.conv3(x_3), conv3], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # out = F.upsample(up5, size=(temp_H, temp_W), mode='bilinear')
        # out = self.gf(feature_guidance, out, feature_guidance,self.attentionblock6(feature_guidance,out))
        # up6 = self.up6(out)
        #
        # feature_guidance = torch.cat([self.conv2(x_2), conv2], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # out = F.upsample(up6, size=(temp_H, temp_W), mode='bilinear')
        # out = self.gf(feature_guidance, out, feature_guidance,self.attentionblock7(feature_guidance,out))
        # up7 = self.up7(out)
        #
        # feature_guidance = torch.cat([conv1, conv1], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # out = F.upsample(up7, size=(temp_H, temp_W), mode='bilinear')
        # out = self.gf(feature_guidance, out, feature_guidance,self.attentionblock8(feature_guidance,out))
        # up8 = self.up8(out)



        feature_guidance = torch.cat([self.conv4(x_4), conv4], dim=1)
        temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        out = self.gf(feature_guidance_small, out, feature_guidance,self.attentionblock5(feature_guidance_small,out))
        up5 = self.up5(out)

        feature_guidance = torch.cat([self.conv3(x_3), conv3], dim=1)
        temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        out = self.gf(feature_guidance_small, up5, feature_guidance,self.attentionblock6(feature_guidance_small,up5))
        up6 = self.up6(out)

        feature_guidance = torch.cat([self.conv2(x_2), conv2], dim=1)
        temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        out = self.gf(feature_guidance_small, up6, feature_guidance,self.attentionblock7(feature_guidance_small,up6))
        up7 = self.up7(out)

        feature_guidance = torch.cat([conv1, conv1], dim=1)
        temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        out = self.gf(feature_guidance_small, up7, feature_guidance,self.attentionblock8(feature_guidance_small,up7))
        up8 = self.up8(out)



        # feature_guidance = torch.cat([self.conv4(x_4), conv4], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        # out = self.gf(feature_guidance_small, out, feature_guidance,self.attentionblock5(feature_guidance_small,out_gate))
        # up5 = self.up5(out)
        #
        # feature_guidance = torch.cat([self.conv3(x_3), conv3], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        # out = self.gf(feature_guidance_small, up5, feature_guidance,self.attentionblock6(feature_guidance_small,out_gate))
        # up6 = self.up6(out)
        #
        # feature_guidance = torch.cat([self.conv2(x_2), conv2], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        # out = self.gf(feature_guidance_small, up6, feature_guidance,self.attentionblock7(feature_guidance_small,out_gate))
        # up7 = self.up7(out)
        #
        # feature_guidance = torch.cat([conv1, conv1], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        # out = self.gf(feature_guidance_small, up7, feature_guidance,self.attentionblock8(feature_guidance_small,out_gate))
        # up8 = self.up8(out)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class G_MM_1_AT(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_MM_1_AT, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter_attention(r=2,eps=0.01)

        # attention blocks

        self.attentionblock5 = GridAttentionBlock(in_channels=512)
        self.attentionblock6 = GridAttentionBlock(in_channels=256)
        self.attentionblock7 = GridAttentionBlock(in_channels=128)
        self.attentionblock8 = GridAttentionBlock(in_channels=64)

        # self.attentionblock5 = GridAttentionBlock_2(in_channels=512,gating_channels=512)
        # self.attentionblock6 = GridAttentionBlock_2(in_channels=256,gating_channels=512)
        # self.attentionblock7 = GridAttentionBlock_2(in_channels=128,gating_channels=512)
        # self.attentionblock8 = GridAttentionBlock_2(in_channels=64,gating_channels=512)

        self.attentionblock1 = GridAttentionBlock_2(in_channels=1,gating_channels=64)
        self.attentionblock2 = GridAttentionBlock_2(in_channels=1,gating_channels=64)
        self.attentionblock3 = GridAttentionBlock_2(in_channels=1,gating_channels=64)
        self.attentionblock4 = GridAttentionBlock_2(in_channels=1,gating_channels=64)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)


        # feature_guidance = torch.cat([self.conv4(x_4), conv4], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # out = F.upsample(out, size=(temp_H, temp_W), mode='bilinear')
        # out = self.gf(feature_guidance, out, feature_guidance,self.attentionblock5(feature_guidance,out))
        # up5 = self.up5(out)
        #
        # feature_guidance = torch.cat([self.conv3(x_3), conv3], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # out = F.upsample(up5, size=(temp_H, temp_W), mode='bilinear')
        # out = self.gf(feature_guidance, out, feature_guidance,self.attentionblock6(feature_guidance,out))
        # up6 = self.up6(out)
        #
        # feature_guidance = torch.cat([self.conv2(x_2), conv2], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # out = F.upsample(up6, size=(temp_H, temp_W), mode='bilinear')
        # out = self.gf(feature_guidance, out, feature_guidance,self.attentionblock7(feature_guidance,out))
        # up7 = self.up7(out)
        #
        # feature_guidance = torch.cat([conv1, conv1], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # out = F.upsample(up7, size=(temp_H, temp_W), mode='bilinear')
        # out = self.gf(feature_guidance, out, feature_guidance,self.attentionblock8(feature_guidance,out))
        # up8 = self.up8(out)




        feature_guidance = torch.cat([self.conv4(x_4), conv4], dim=1)
        temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        out = self.gf(feature_guidance_small, out, feature_guidance,self.attentionblock5(feature_guidance_small,out))
        up5 = self.up5(out)

        feature_guidance = torch.cat([self.conv3(x_3), conv3], dim=1)
        temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        out = self.gf(feature_guidance_small, up5, feature_guidance,self.attentionblock6(feature_guidance_small,up5))
        up6 = self.up6(out)

        feature_guidance = torch.cat([self.conv2(x_2), conv2], dim=1)
        temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        out = self.gf(feature_guidance_small, up6, feature_guidance,self.attentionblock7(feature_guidance_small,up6))
        up7 = self.up7(out)

        feature_guidance = torch.cat([conv1, conv1], dim=1)
        temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        out = self.gf(feature_guidance_small, up7, feature_guidance,self.attentionblock8(feature_guidance_small,up7))
        up8 = self.up8(out)

        out_gate = out
        # # bottom feature as gate
        # out_gate = out
        # feature_guidance = torch.cat([self.conv4(x_4), conv4], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        # out = self.gf(feature_guidance_small, out, feature_guidance,self.attentionblock5(feature_guidance_small,out_gate))
        # up5 = self.up5(out)
        # feature_guidance = torch.cat([self.conv3(x_3), conv3], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        # out = self.gf(feature_guidance_small, up5, feature_guidance,self.attentionblock6(feature_guidance_small,out_gate))
        # up6 = self.up6(out)
        # feature_guidance = torch.cat([self.conv2(x_2), conv2], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        # out = self.gf(feature_guidance_small, up6, feature_guidance,self.attentionblock7(feature_guidance_small,out_gate))
        # up7 = self.up7(out)
        # feature_guidance = torch.cat([conv1, conv1], dim=1)
        # temp_N, temp_C, temp_H, temp_W = feature_guidance.size()
        # feature_guidance_small = F.upsample(feature_guidance, size=(temp_H/2, temp_W/2), mode='bilinear')
        # out = self.gf(feature_guidance_small, up7, feature_guidance,self.attentionblock8(feature_guidance_small,out_gate))
        # up8 = self.up8(out)

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)

        side_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h), self.attentionblock1(self.guided_map(x_4), out_gate))
        side_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h), self.attentionblock2(self.guided_map(x_3), out_gate))
        side_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h), self.attentionblock3(self.guided_map(x_2), out_gate))
        side_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h), self.attentionblock4(self.guided_map(x), out_gate))

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Multi_1(nn.Module):
    """
    compare Guided Filter with skip-connection and upsample
    author: Shihao Zhang
    Data: 2019/1/22
    """
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Multi_1, self).__init__()

        # mutli-scale simple convolution
        # self.conv0 = M_Conv(3, 16, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv1 = M_Conv(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv2 = M_Conv(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_4(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_4(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_4(64 , 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.side_9 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.gf = FastGuidedFilter(r=2, eps=0.01)
        self.gf1 = FastGuidedFilter(r=2, eps=0.8)

    def forward(self, x,x_h):
        _, _, img_shape, _ = x.size()
        _, _, img_shape_filter, _ = x_h.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')

        x_h = self.conv1(x_h)
        # x = self.conv1(x)
        x = F.upsample(x_h, size=(img_shape, img_shape), mode='bilinear')
        conv1, out = self.down1(x)

        x_2 = self.conv2(x_2)
        x_2 = torch.cat([x_2, F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')], dim=1)
        out = torch.cat([x_2, out], dim=1)
        conv2, out = self.down2(out)

        x_3 = self.conv3(x_3)
        x_3 = torch.cat([x_3, F.upsample(x_2, size=(img_shape / 4, img_shape / 4), mode='bilinear')], dim=1)
        out = torch.cat([x_3, out], dim=1)
        conv3, out = self.down3(out)

        x_4 = self.conv4(x_4)
        x_4 = torch.cat([x_4, F.upsample(x_3, size=(img_shape / 8, img_shape / 8), mode='bilinear')], dim=1)
        out = torch.cat([x_4, out], dim=1)
        conv4, out = self.down4(out)

        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv4, up5,x_3,conv3)
        up7 = self.up7(conv3, up6,x_2,conv2)
        up8 = self.up8(conv2, up7,x,conv1)

        # up8 = self.gf(conv1,up8,conv1)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')



        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        x_h = self.gf1(x,conv1,x_h)
        x_h = self.gf(conv1,up8,x_h)

        # x_h_0 = self.gf(x,conv1,x_h)
        # x_h_0 = self.gf(conv1,up8,x_h)
        #
        # x_h_1 = self.gf(x_4[:,-16:,:,:], up5[:,-16:,:,:], x_h)
        # x_h_2 = self.gf(x_3[:,-16:,:,:], up6[:,-16:,:,:], x_h)
        # x_h_2 = self.gf(x_2[:,-16:,:,:], up7[:,-16:,:,:], x_h)
        # x_h = torch.cat([ self.gf(x,up8,x_h),self.gf(x_4[:,-32:,:,:], up5[:,-32:,:,:], x_h),self.gf(x_3[:,-32:,:,:], up6[:,-32:,:,:], x_h),self.gf(x_2[:,-32:,:,:], up7[:,-32:,:,:], x_h)],dim = 1)


        # x_h = a[:,-16:,:,:] * x_h+b[:,-16:,:,:]
        # x_h = self.side_9(x_h)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)

        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        return [ave_out, side_5, side_6, side_7, side_8]
        # return [ave_out, side_5, side_6, side_7, side_8, x_h]

# ----------------------------------------------------------
# Adding super-resolution into models
# ----------------------------------------------------------

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()

        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        output = self.cov_block(x)
        return output


class _Conv_Block_My(nn.Module):
    def __init__(self, n_channel=64, out_channel=64):
        super(_Conv_Block_My, self).__init__()

        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=n_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        output = self.cov_block(x)
        return output

class _Conv_Block_My1(nn.Module):
    def __init__(self, n_channel=64, out_channel=64):
        super(_Conv_Block_My1, self).__init__()

        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=n_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        output = self.cov_block(x)
        return output

class SR_M(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(SR_M, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the SR(super-resolution)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv_input1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_I1 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block)


        self.conv_input2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_I2 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block)


        self.conv_input3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_I3 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_F3 = self.make_layer(_Conv_Block)


        self.conv_input4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_I4 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_F4 = self.make_layer(_Conv_Block)


        # the Guided Filter
        self.gf = FastGuidedFilter(radius, eps)
        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1)
        )
        self.guided_map.apply(weights_init_identity)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)



    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)

        x = self.convt_I1(x)
        x_2 = self.convt_I2(x_2)
        x_3 = self.convt_I3(x_3)
        x_4 = self.convt_I4(x_4)

        up5 = self.relu(self.conv_input1(up5))
        up5 = self.convt_F1(up5)

        h5 = x_4 + up5

        up6 = self.relu(self.conv_input2(up6))
        up6 = self.convt_F1(up6)

        h6 = x_3 + up6

        up7 = self.relu(self.conv_input3(up7))
        up7 = self.convt_F1(up7)

        h7 = x_2 + up7

        up8 = self.relu(self.conv_input4(up8))
        up8 = self.convt_F1(up8)

        h8 = x + up8

        up5 = self.guided_map(up5)
        up6 = self.guided_map(up6)
        up7 = self.guided_map(up7)
        up8 = self.guided_map(up8)

        side_5 = self.gf(up5, F.upsample(side_5, size=(img_shape / 4, img_shape / 4), mode='bilinear'), up5)
        side_5 = self.gf(up6, F.upsample(side_5, size=(img_shape / 2, img_shape / 2), mode='bilinear'), up6)
        side_5 = self.gf(up7, F.upsample(side_5, size=(img_shape, img_shape ), mode='bilinear'), up7)
        side_5 = self.gf(up8, F.upsample(side_5, size=(img_shape * 2, img_shape * 2), mode='bilinear'), up8)

        side_6 = self.gf(up6, F.upsample(side_6, size=(img_shape / 2, img_shape / 2), mode='bilinear'), up6)
        side_6 = self.gf(up7, F.upsample(side_6, size=(img_shape, img_shape ), mode='bilinear'), up7)
        side_6 = self.gf(up8, F.upsample(side_6, size=(img_shape * 2, img_shape * 2), mode='bilinear'), up8)

        side_7 = self.gf(up7, F.upsample(side_7, size=(img_shape, img_shape ), mode='bilinear'), up7)
        side_7 = self.gf(up8, F.upsample(side_7, size=(img_shape * 2, img_shape * 2), mode='bilinear'), up8)

        side_8 = self.gf(up8, F.upsample(side_8, size=(img_shape * 2, img_shape * 2), mode='bilinear'), up8)

        ave_out = (side_5+side_6+side_7+side_8)/4

        h5 = F.upsample(h5, size=(img_shape*2, img_shape*2), mode='bilinear')
        h6 = F.upsample(h6, size=(img_shape*2, img_shape*2), mode='bilinear')
        h7 = F.upsample(h7, size=(img_shape*2, img_shape*2), mode='bilinear')
        h8 = F.upsample(h8, size=(img_shape*2, img_shape*2), mode='bilinear')


        return [ave_out, side_5, side_6, side_7, side_8, h5, h6, h7, h8]


class SR_M_Resdual(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(SR_M_Resdual, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the resdual
        self.side_center = nn.Conv2d(512, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the SR(super-resolution)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv_input1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_I1 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block)


        self.conv_input2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_I2 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block)


        self.conv_input3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_I3 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_F3 = self.make_layer(_Conv_Block)


        self.conv_input4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_I4 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_F4 = self.make_layer(_Conv_Block)

        self.conv_input_center = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F_center = self.make_layer(_Conv_Block)


        # the Guided Filter
        self.gf = FastGuidedFilter(radius, eps)
        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1)
        )
        self.guided_map.apply(weights_init_identity)


        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)

        side_center = self.side_center(out)
        sr_center = self.relu(self.conv_input_center(out))
        sr_center = self.convt_F_center(sr_center)

        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        side_center = self.gf(self.guided_map(sr_center), F.upsample(side_center, size=(img_shape / 8, img_shape / 8), mode='bilinear'), self.guided_map(sr_center))

        h_5 = self.relu(self.conv_input1(up5))
        h_5 = self.convt_F1(h_5)
        h_5 = self.convt_I1(sr_center) + h_5
        side_5 = self.side_5(up5) + side_center
        side_5 = self.gf(self.guided_map(h_5), F.upsample(side_5, size=(img_shape / 4, img_shape / 4), mode='bilinear'), self.guided_map(h_5))

        h_6 = self.relu(self.conv_input2(up6))
        h_6 = self.convt_F2(h_6)
        h_6 = self.convt_I2(h_5) + h_6
        side_6 = self.side_6(up6) + side_5
        side_6 = self.gf(self.guided_map(h_6), F.upsample(side_6, size=(img_shape / 2, img_shape / 2), mode='bilinear'), self.guided_map(h_6))

        h_7 = self.relu(self.conv_input3(up7))
        h_7 = self.convt_F3(h_7)
        h_7 = self.convt_I3(h_6) + h_7
        side_7 = self.side_7(up7) + side_6
        side_7 = self.gf(self.guided_map(h_7), F.upsample(side_7, size=(img_shape , img_shape ), mode='bilinear'), self.guided_map(h_7))

        h_8 = self.relu(self.conv_input4(up8))
        h_8 = self.convt_F4(h_8)
        h_8 = self.convt_I4(h_7) + h_8
        side_8 = self.side_8(up8) + side_7
        side_8 = self.gf(self.guided_map(h_8), F.upsample(side_8, size=(img_shape * 2, img_shape * 2), mode='bilinear'), self.guided_map(h_8))

        side_5 = F.upsample(side_5, size= (img_shape * 2, img_shape * 2), mode='bilinear')
        side_6 = F.upsample(side_6, size= (img_shape * 2, img_shape * 2), mode='bilinear')
        side_7 = F.upsample(side_7, size= (img_shape * 2, img_shape * 2), mode='bilinear')
        side_8 = F.upsample(side_8, size= (img_shape * 2, img_shape * 2), mode='bilinear')
        side_center = F.upsample(side_center, size= (img_shape * 2, img_shape * 2), mode='bilinear')
        #
        # h_5 = F.upsample(h_5, size=(img_shape*2, img_shape*2), mode='bilinear')
        # h_6 = F.upsample(h_6, size=(img_shape*2, img_shape*2), mode='bilinear')
        # h_7 = F.upsample(h_7, size=(img_shape*2, img_shape*2), mode='bilinear')
        # h_8 = F.upsample(h_8, size=(img_shape*2, img_shape*2), mode='bilinear')
        # sr_center = F.upsample(sr_center, size=(img_shape*2, img_shape*2), mode='bilinear')

        ave_out = (side_5+side_6+side_7+side_8)/4

        return [ave_out, side_5, side_6, side_7, side_8, h_5, h_6, h_7, h_8, side_center, sr_center]


class SR_M_1(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(SR_M_1, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the SR(super-resolution)
        self.gf = FastGuidedFilter(r=2, eps=0.01)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.convt_F1 = self.make_layer(_Conv_Block_My, 32)
        self.convt_R1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)

    def make_layer(self, block, n_channel=64):
        layers = []
        layers.append(block(n_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        sr = self.convt_F1(conv1)
        sr = self.convt_R1(sr) + self.convt_I1(torch.unsqueeze(x[:,0,:,:],0))


        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)

        side_5 = self.gf(sr,F.upsample(side_5, size=(img_shape * 2, img_shape * 2), mode='bilinear'),sr)
        side_6 = self.gf(sr,F.upsample(side_6, size=(img_shape * 2, img_shape * 2), mode='bilinear'),sr)
        side_7 = self.gf(sr,F.upsample(side_7, size=(img_shape * 2, img_shape * 2), mode='bilinear'),sr)
        side_8 = self.gf(sr,F.upsample(side_8, size=(img_shape * 2, img_shape * 2), mode='bilinear'),sr)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        return [ave_out, side_5, side_6, side_7, side_8, sr]


class SR_M_2(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(SR_M_2, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the SR(super-resolution)
        self.gf = FastGuidedFilter(r=2, eps=0.01)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.convt_F1 = self.make_layer(_Conv_Block_My, 32)
        self.convt_R1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)

    def make_layer(self, block, n_channel=64):
        layers = []
        layers.append(block(n_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        sr = self.convt_F1(conv1)
        sr = self.convt_R1(sr) + self.convt_I1(torch.unsqueeze(x[:,0,:,:],0))

        side_5 = self.gf(sr,F.upsample(up5, size=(img_shape * 2, img_shape * 2), mode='bilinear'),sr)
        side_6 = self.gf(sr,F.upsample(up6, size=(img_shape * 2, img_shape * 2), mode='bilinear'),sr)
        side_7 = self.gf(sr,F.upsample(up7, size=(img_shape * 2, img_shape * 2), mode='bilinear'),sr)
        side_8 = self.gf(sr,F.upsample(up8, size=(img_shape * 2, img_shape * 2), mode='bilinear'),sr)

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        return [ave_out, side_5, side_6, side_7, side_8, sr]


class SR_M_3(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(SR_M_3, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the SR(super-resolution)
        self.gf = FastGuidedFilter(r=2, eps=0.01)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.convt_F5 = self.make_layer(_Conv_Block_My, 256, 256)
        self.convt_R5 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input5 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.convt_F6 = self.make_layer(_Conv_Block_My, 256, 128)
        self.convt_R6 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input6 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.convt_F7 = self.make_layer(_Conv_Block_My,128, 64)
        self.convt_R7 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input7 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

        self.convt_F8 = self.make_layer(_Conv_Block_My, 64, 32)
        self.convt_R8 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input8 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

    def make_layer(self, block, n_channel=64, out_channel = 64):
        layers = []
        layers.append(block(n_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)

        out = self.relu(self.conv_input5(x_4))
        h_5 = self.convt_F5(out)
        up5 = self.gf(h_5, F.upsample(up5, size=(img_shape / 4, img_shape / 4), mode='bilinear'), h_5)
        h_5 = self.convt_R5(h_5)

        up6 = self.up6(conv3, up5)
        out = self.relu(self.conv_input6(x_3))
        out = self.gf(up5, out, up5)
        h_6 = self.convt_F6(out)
        up6 = self.gf(h_6, F.upsample(up6, size=(img_shape / 2, img_shape / 2), mode='bilinear'), h_6)
        h_6 = self.convt_R6(h_6)

        up7 = self.up7(conv2, up6)
        out = self.relu(self.conv_input7(x_2))
        out = self.gf(up6, out, up6)
        h_7 = self.convt_F7(out)
        up7 = self.gf(h_7, F.upsample(up7, size=(img_shape, img_shape), mode='bilinear'), h_7)
        h_7 = self.convt_R7(h_7)

        up8 = self.up8(conv1, up7)
        out = self.relu(self.conv_input8(x))
        out = self.gf(up7, out, up7)
        h_8 = self.convt_F8(out)
        up8 = self.gf(h_8, F.upsample(up8, size=(img_shape * 2, img_shape * 2), mode='bilinear'), h_8)
        h_8 = self.convt_R8(h_8)

        side_5 = F.upsample(up5, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape * 2, img_shape * 2), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        # ave_out = side_8
        return [ave_out, side_5, side_6, side_7, side_8, h_5, h_6, h_7, h_8]


class SR_M_4(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(SR_M_4, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the SR(super-resolution)
        self.gf = FastGuidedFilter(r=2, eps=0.01)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.convt_F1 = self.make_layer(_Conv_Block_My, 64)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_input1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)


    def make_layer(self, block, n_channel=64, out_channel=64):
        layers = []
        layers.append(block(n_channel,out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        out = self.relu(self.conv_input1(x))
        sr = self.convt_F1(torch.cat([up8, out], dim=1))
        sr = self.convt_R1(sr)

        side_5 = F.upsample(up5, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape * 2, img_shape * 2), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        return [ave_out, side_5, side_6, side_7, side_8, sr, sr, sr, sr]


class SR_M_5(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(SR_M_5, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the SR(super-resolution)
        self.gf = FastGuidedFilter(r=2, eps=0.01)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.convt_F1 = self.make_layer(_Conv_Block_My1, 32, 32)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.T1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv_input1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)


    def make_layer(self, block, n_channel=64, out_channel=64):
        layers = []
        layers.append(block(n_channel,out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        out = self.relu(self.conv_input1(x))
        sr = self.convt_F1(out)
        # side_8 = torch.cat([up8, sr], dim=1)
        # side_8 = up8
        sr = self.T1(torch.cat([up8, sr], dim=1))
        sr = self.convt_R1(sr)

        # sr = torch.unsqueeze(F.upsample(x, size=(img_shape * 2, img_shape * 2), mode='bilinear')[:, 0, :, :], 1)

        side_5 = self.gf(torch.unsqueeze(x_4[:, 0, :, :], 1), up5, sr)
        side_6 = self.gf(torch.unsqueeze(x_3[:, 0, :, :], 1), up6, sr)
        side_7 = self.gf(torch.unsqueeze(x_2[:, 0, :, :], 1), up7, sr)
        side_8 = self.gf(torch.unsqueeze(x[:, 0, :, :], 1), up8, sr)


        # side_5 = F.upsample(up5, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        # side_8 = F.upsample(side_8, size=(img_shape * 2, img_shape * 2), mode='bilinear')

        # side_5 = F.upsample(up5, size=(img_shape, img_shape ), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape ), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape , img_shape ), mode='bilinear')
        # side_8 = F.upsample(side_8, size=(img_shape , img_shape ), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        return [ave_out, side_5, side_6, side_7, side_8, sr, sr, sr, sr]


class SR_M_6(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(SR_M_6, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the SR(super-resolution)
        self.gf = FastGuidedFilter(r=2, eps=0.01)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.convt_F1 = self.make_layer(_Conv_Block_My1, 32, 32)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.T1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv_input1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)


    def make_layer(self, block, n_channel=64, out_channel=64):
        layers = []
        layers.append(block(n_channel,out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # out = self.relu(self.conv_input1(x))
        # sr = self.convt_F1(out)
        # # side_8 = torch.cat([up8, sr], dim=1)
        # # side_8 = up8
        # sr = self.T1(torch.cat([up8, sr], dim=1))
        # sr = self.convt_R1(sr)

        sr = torch.unsqueeze(F.upsample(x, size=(img_shape * 2, img_shape * 2), mode='bilinear')[:, 0, :, :], 1)

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)

        side_5 = self.gf(torch.unsqueeze(x_4[:, 0, :, :], 1), side_5, sr)
        side_6 = self.gf(torch.unsqueeze(x_3[:, 0, :, :], 1), side_6, sr)
        side_7 = self.gf(torch.unsqueeze(x_2[:, 0, :, :], 1), side_7, sr)
        side_8 = self.gf(torch.unsqueeze(x[:, 0, :, :], 1), side_8, sr)

        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        return [ave_out, side_5, side_6, side_7, side_8, sr, sr, sr, sr]


class SR_M_7(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(SR_M_7, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the SR(super-resolution)
        self.gf = FastGuidedFilter(r=2, eps=0.01)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sr_up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.sr_up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.sr_up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.sr_up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        # self.convt_F1 = self.make_layer(_Conv_Block_My1, 32, 32)
        self.convt_R1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.T1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.TImage = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        # self.conv_input1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        sr_up5 = self.sr_up5(conv4, out)
        # sr_up6 = self.sr_up6(conv3, up5)
        # sr_up7 = self.sr_up7(conv2, up6)
        # sr_up8 = self.sr_up8(conv1, up7)
        sr_up6 = self.sr_up6(conv3, sr_up5)
        sr_up7 = self.sr_up7(conv2, sr_up6)
        sr_up8 = self.sr_up8(conv1, sr_up7)
        sr = self.T1(sr_up8)
        # sr = self.convt_R1(sr)+ F.upsample(x, size=(img_shape * 2, img_shape * 2), mode='bilinear')[:,0,:,:]
        sr = self.convt_R1(sr)
        # sr_temp = sr.clone()
        sr_temp = sr.clone()

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.gf(torch.unsqueeze(x_4[:, 0, :, :], 1), up5, sr_temp)
        side_6 = self.gf(torch.unsqueeze(x_3[:, 0, :, :], 1), up6, sr_temp)
        side_7 = self.gf(torch.unsqueeze(x_2[:, 0, :, :], 1), up7, sr_temp)
        side_8 = self.gf(torch.unsqueeze(x[:, 0, :, :], 1), up8, sr_temp)

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        return [ave_out, side_5, side_6, side_7, side_8, sr, sr, sr, sr]


class SR_M_8(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(SR_M_8, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the SR(super-resolution)
        self.gf = FastGuidedFilter(r=2, eps=0.01)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sr_up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.sr_up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.sr_up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.sr_up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        # self.convt_F1 = self.make_layer(_Conv_Block_My1, 32, 32)
        self.convt_R1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.T1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.TImage = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        # self.conv_input1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)

        self.convt_R2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.T2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        sr_up5 = self.sr_up5(conv4, out)
        sr_up6 = self.sr_up6(conv3, sr_up5)
        sr_up7 = self.sr_up7(conv2, sr_up6)
        sr_up8 = self.sr_up8(conv1, sr_up7)
        sr = self.T1(sr_up8)
        sr = self.convt_R1(sr)

        sr_guided = self.T2(sr_up8)
        sr_guided = self.convt_R2(sr_guided)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.gf(torch.unsqueeze(x_4[:, 0, :, :], 1), up5, sr_guided)
        side_6 = self.gf(torch.unsqueeze(x_3[:, 0, :, :], 1), up6, sr_guided)
        side_7 = self.gf(torch.unsqueeze(x_2[:, 0, :, :], 1), up7, sr_guided)
        side_8 = self.gf(torch.unsqueeze(x[:, 0, :, :], 1), up8, sr_guided)



        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        return [ave_out, side_5, side_6, side_7, side_8, sr, sr, sr, sr]


class G_Up_temp(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_Up_temp, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the SR(super-resolution)
        self.gf = FastGuidedFilter(r=2, eps=0.01)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.convt_F1 = self.make_layer(_Conv_Block_My1, 32, 32)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.T1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv_input1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)


    def make_layer(self, block, n_channel=64, out_channel=64):
        layers = []
        layers.append(block(n_channel,out_channel))
        return nn.Sequential(*layers)

    def forward(self, x,x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # out = self.relu(self.conv_input1(x))
        # sr = self.convt_F1(out)
        # # side_8 = torch.cat([up8, sr], dim=1)
        # # side_8 = up8
        # sr = self.T1(torch.cat([up8, sr], dim=1))
        # sr = self.convt_R1(sr)

        sr = torch.unsqueeze(x_h[:, 0, :, :], 1)

        side_5 = self.gf(torch.unsqueeze(x_4[:, 0, :, :], 1), up5, sr)
        side_6 = self.gf(torch.unsqueeze(x_3[:, 0, :, :], 1), up6, sr)
        side_7 = self.gf(torch.unsqueeze(x_2[:, 0, :, :], 1), up7, sr)
        side_8 = self.gf(torch.unsqueeze(x[:, 0, :, :], 1), up8, sr)


        # side_5 = F.upsample(up5, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        # side_8 = F.upsample(side_8, size=(img_shape * 2, img_shape * 2), mode='bilinear')

        # side_5 = F.upsample(up5, size=(img_shape, img_shape ), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape ), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape , img_shape ), mode='bilinear')
        # side_8 = F.upsample(side_8, size=(img_shape , img_shape ), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        return [ave_out, side_5, side_6, side_7, side_8]


class SR_M_only_sr(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(SR_M_only_sr, self).__init__()

        self.conv_seg = nn.Conv2d(in_channels=3, out_channels=n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        # the SR(super-resolution)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.convt_F1 = self.make_layer(_Conv_Block_My, 64)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

    def make_layer(self, block, n_channel=64):
        layers = []
        layers.append(block(n_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, img_shape, _ = x.size()

        x = F.upsample(x, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        # _, _, img_shape, _ = x.size()

        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')

        out = self.relu(self.conv_input(x))
        sr = self.convt_F1(out)
        # sr = self.convt_R1(sr) + self.convt_I1(torch.unsqueeze(x[:,0,:,:],0))
        sr = self.convt_R1(sr)

        # out = self.relu(self.conv_input(x_4))
        # h_5 = self.convt_F1(out)
        # # h_5 = self.convt_R1(h_5) + self.convt_I1(torch.unsqueeze(x_4[:,0,:,:],0))
        # h_5 = self.convt_R1(h_5)
        #
        # out = self.relu(self.conv_input(x_3))
        # h_6 = self.convt_F1(out)
        # # h_6 = self.convt_R1(h_6) + self.convt_I1(torch.unsqueeze(x_3[:,0,:,:],0))
        # h_6 = self.convt_R1(h_6)
        #
        # out = self.relu(self.conv_input(x_2))
        # h_7 = self.convt_F1(out)
        # # h_7 = self.convt_R1(h_7) + self.convt_I1(torch.unsqueeze(x_2[:,0,:,:],0))
        # h_7 = self.convt_R1(h_7)

        out_seg = self.conv_seg(x)
        # out_seg = F.upsample(out_seg, size=(512, 512))
        return [out_seg, out_seg, out_seg, out_seg, out_seg, sr, sr, sr, sr]
        # return [out_seg, out_seg, out_seg, out_seg, out_seg, h_5, h_6, h_7, sr]


class SR_M_only_sr2(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(SR_M_only_sr2, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the SR(super-resolution)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.convt_F1 = self.make_layer(_Conv_Block_My, 32, 32)
        self.convt_R1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)

    def make_layer(self, block, n_channel=64, out_channel=64):
        layers = []
        layers.append(block(n_channel,out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, img_shape, _ = x.size()

        x = F.upsample(x, size=(img_shape * 2, img_shape * 2), mode='bilinear')
        _, _, img_shape, _ = x.size()

        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # out = self.relu(self.conv_input(x))
        # sr = self.convt_F1(torch.cat([up8, out], dim=1))
        # sr = self.convt_R1(sr)

        # sr = self.convt_F1(torch.cat([up8, out], dim=1))
        # sr = self.convt_R1(sr)

        sr = self.convt_R1(up8)

        out_seg = self.side_8(up8)
        # out_seg = F.upsample(out_seg, size=(512, 512))



        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        return [out_seg, out_seg, out_seg, out_seg, out_seg, sr, sr, sr, sr]


class M_Net_Resdual(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Resdual, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # the resdual
        self.convt_T5 = nn.ConvTranspose2d(in_channels=n_classes, out_channels=n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_T6 = nn.ConvTranspose2d(in_channels=n_classes, out_channels=n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_T7 = nn.ConvTranspose2d(in_channels=n_classes, out_channels=n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_T8 = nn.ConvTranspose2d(in_channels=n_classes, out_channels=n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.side_center = nn.Conv2d(512, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)

        side_center = self.side_center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5) + self.convt_T5(side_center)
        side_6 = self.side_6(up6) + self.convt_T6(side_5)
        side_7 = self.side_7(up7) + self.convt_T7(side_6)
        side_8 = self.side_8(up8) + self.convt_T8(side_7)

        side_5 = F.upsample(side_5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(side_6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(side_7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(side_8, size=(img_shape, img_shape), mode='bilinear')

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_Resdual_Up_8(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_Resdual_Up_8, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.gf = FastGuidedFilter(r=2,eps=0.01)

        # the resdual
        self.convt_T5 = nn.ConvTranspose2d(in_channels=n_classes, out_channels=n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_T6 = nn.ConvTranspose2d(in_channels=n_classes, out_channels=n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_T7 = nn.ConvTranspose2d(in_channels=n_classes, out_channels=n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_T8 = nn.ConvTranspose2d(in_channels=n_classes, out_channels=n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.side_center = nn.Conv2d(512, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        side_center = self.side_center(out)

        up5 = self.up5(conv4, out)
        up5 = self.gf(conv4, up5, conv4)

        up6 = self.up6(conv3, up5)
        up6 = self.gf(conv3, up6, conv3)

        up7 = self.up7(conv2, up6)
        up7 = self.gf(conv2, up7, conv2)

        up8 = self.up8(conv1, up7)
        up8 = self.gf(conv1, up8, conv1)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5) + self.convt_T5(side_center)
        side_6 = self.side_6(up6) + self.convt_T6(side_5)
        side_7 = self.side_7(up7) + self.convt_T7(side_6)
        side_8 = self.side_8(up8) + self.convt_T8(side_7)

        side_5 = F.upsample(side_5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(side_6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(side_7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(side_8, size=(img_shape, img_shape), mode='bilinear')

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class GM(nn.Module):
    """
    Guided Filter with M-Net
    Author: Shihao Zhang
    Time: 2018/10/26
    """
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(GM, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter(radius, eps)
        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x,x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5+side_6+side_7+side_8)/4

        ave_out = self.gf(self.guided_map(x), ave_out, self.guided_map(x_h))
        return [ave_out, side_5, side_6, side_7, side_8]


class GF(nn.Module):
    """
    Guided Filter using figure with M-Net
    Author: Shihao Zhang
    Time: 2018/10/26
    """
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(GF, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter(radius, eps)
        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, cn, 1, bias=False),
        #     AdaptiveNorm(cn),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(cn, 1, 1)
        # )
        # self.guided_map.apply(weights_init_identity)

    def forward(self, x,x_l,x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5+side_6+side_7+side_8)/4

        x_h=torch.unsqueeze(x_h,1)
        x_l = torch.unsqueeze(x_l, 1)

        ave_out = self.gf(x_l, ave_out, x_h)
        return [ave_out, side_5, side_6, side_7, side_8]


class G_MM(nn.Module):
    """
    Mulit Guided Filter with M-Net
    Author: Shihao Zhang
    Time: 2018/10/26
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_MM, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter(radius, eps)
        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)
        # print(side_5.shape)
        # print(x_4.shape)
        # print(self.guided_map(x_4).shape)
        # print('------')

        side_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
        side_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
        side_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
        side_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4

        # ave_out = self.gf(self.guided_map(x), ave_out, self.guided_map(x_h))
        return [ave_out, side_5, side_6, side_7, side_8]


class G_MM_1(nn.Module):
    """
    Mulit Guided Filter with M-Net
    Author: Shihao Zhang
    Time: 2018/10/26
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_MM_1, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter(radius, eps)

        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, cn, 1, bias=False),
        #     AdaptiveNorm(cn),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(cn, 1, 1)
        # )

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, 1, 1, bias=False),
        #     AdaptiveNorm(1)
        # )
        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(1, 1, kernel_size=1, padding=0, stride=1, bias=True),
        #     AdaptiveNorm(1)
        # )
        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1, bias=True),
        #     AdaptiveNorm(1)
        # )
        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, 1, kernel_size=1, padding=0, stride=1, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=True),
        #     AdaptiveNorm(1)
        # )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)
        # print(side_5.shape)
        # print(x_4.shape)
        # print(self.guided_map(x_4).shape)
        # print('------')

        side_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
        side_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
        side_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
        side_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4

        ave_out = self.gf(self.guided_map(x_h), ave_out, self.guided_map(x_h))
        return [ave_out, side_5, side_6, side_7, side_8]


class G_MM_New_1(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_MM_New_1, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_5(512 , 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_5(256 , 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_5(128 , 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter(radius, eps)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)

        x_2_ = self.conv2(x_2)
        out = torch.cat([x_2_, out], dim=1)
        conv2, out = self.down2(out)

        x_3_ = self.conv3(x_3)
        out = torch.cat([x_3_, out], dim=1)
        conv3, out = self.down3(out)

        x_4_ = self.conv4(x_4)
        out = torch.cat([x_4_, out], dim=1)
        conv4, out = self.down4(out)

        out = self.center(out)
        up5 = self.up5(conv4, out, x_4_)
        up6 = self.up6(conv3, up5, x_3_)
        up7 = self.up7(conv2, up6, x_2_)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)

        side_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
        side_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
        side_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
        side_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class G_MM_New_2(nn.Module):
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_MM_New_2, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_5(512 , 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_5(256 , 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_5(128 , 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter(radius, eps)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 3, 1),
            AdaptiveNorm(3)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)

        x_2_ = self.conv2(x_2)
        out = torch.cat([x_2_, out], dim=1)
        conv2, out = self.down2(out)

        x_3_ = self.conv3(x_3)
        out = torch.cat([x_3_, out], dim=1)
        conv3, out = self.down3(out)

        x_4_ = self.conv4(x_4)
        out = torch.cat([x_4_, out], dim=1)
        conv4, out = self.down4(out)

        out = self.center(out)
        up5 = self.up5(conv4, out, x_4_)
        up6 = self.up6(conv3, up5, x_3_)
        up7 = self.up7(conv2, up6, x_2_)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)

        side_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
        side_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
        side_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
        side_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class G_MM_compare(nn.Module):
    """
    Mulit Guided Filter with M-Net
    Author: Shihao Zhang
    Time: 2018/10/26
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_MM_compare, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter(radius, eps)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )

        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)

        side_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
        side_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
        side_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
        side_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))

        ave_out = (side_5 + side_6 + side_7 + side_8) / 4

        ave_out = self.gf(self.guided_map(x_h), ave_out, self.guided_map(x_h))
        return [ave_out, side_5, side_6, side_7, side_8]


class G_HuaZhu(nn.Module):
    """
    Modify G_MM_1 model to increase innovation
    Motivated by HuaZhu Fu
    Author: Shihao Zhang
    Time: 2019/1/11
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_HuaZhu, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.combine1 = nn.Conv2d(12, 3, kernel_size=3, padding=1, stride=1, bias=True)
        self.combine1_ = nn.Conv2d(3, 3, kernel_size=1, padding=0, stride=1, bias=True)
        self.combine2 = nn.Conv2d(12, 3, kernel_size=3, padding=1, stride=1, bias=True)
        self.combine2_ = nn.Conv2d(3, 3, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter_my(radius, eps)

        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, cn, 1, bias=False),
        #     AdaptiveNorm(cn),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(cn, 1, 1)
        # )

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)
        # print(side_5.shape)
        # print(x_4.shape)
        # print(self.guided_map(x_4).shape)
        # print('------')

        a_5, b_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
        a_6, b_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
        a_7, b_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
        a_8, b_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))
        a_final = torch.cat([a_5, a_6, a_7, a_8], dim=1)
        b_final = torch.cat([a_5, a_6, a_7, a_8], dim=1)

        a_final = a_final.float()
        b_final = b_final.float()
        a_final = self.combine1(a_final)
        a_final = self.combine1_(a_final)
        b_final = self.combine2(b_final)
        b_final = self.combine2_(b_final)

        return a_final * x_h + b_final


class G_HuaZhu_2(nn.Module):
    """
    Modify G_MM_1 model to increase innovation
    Motivated by HuaZhu Fu
    Author: Shihao Zhang
    Time: 2019/1/12
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_HuaZhu_2, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.combine_a_0 = nn.Conv2d(4, 1, kernel_size=1, padding=0, stride=1, bias=True)
        self.combine_a_1 = nn.Conv2d(4, 1, kernel_size=1, padding=0, stride=1, bias=True)
        self.combine_a_2 = nn.Conv2d(4, 1, kernel_size=1, padding=0, stride=1, bias=True)
        self.combine_b_0 = nn.Conv2d(4, 1, kernel_size=1, padding=0, stride=1, bias=True)
        self.combine_b_1 = nn.Conv2d(4, 1, kernel_size=1, padding=0, stride=1, bias=True)
        self.combine_b_2 = nn.Conv2d(4, 1, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter_my(radius, eps)

        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, cn, 1, bias=False),
        #     AdaptiveNorm(cn),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(cn, 1, 1)
        # )

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)
        # print(side_5.shape)
        # print(x_4.shape)
        # print(self.guided_map(x_4).shape)
        # print('------')

        a_5, b_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
        a_6, b_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
        a_7, b_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
        a_8, b_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))
        temp = a_5[:,0,:,:]
        a_final_0 = self.combine_a_0(torch.cat([torch.unsqueeze(a_5[:,0,:,:],1), torch.unsqueeze(a_6[:,0,:,:],1), torch.unsqueeze(a_7[:,0,:,:],1), torch.unsqueeze(a_8[:,0,:,:],1)], dim=1))
        a_final_1 = self.combine_a_1(torch.cat([torch.unsqueeze(a_5[:,1,:,:],1), torch.unsqueeze(a_6[:,1,:,:],1), torch.unsqueeze(a_7[:,1,:,:],1), torch.unsqueeze(a_8[:,1,:,:],1)], dim=1))
        a_final_2 = self.combine_a_2(torch.cat([torch.unsqueeze(a_5[:,2,:,:],1), torch.unsqueeze(a_6[:,2,:,:],1), torch.unsqueeze(a_7[:,2,:,:],1), torch.unsqueeze(a_8[:,2,:,:],1)], dim=1))
        a_final = torch.cat([a_final_0, a_final_1, a_final_2], dim=1)
        b_final_0 = self.combine_b_0(torch.cat([torch.unsqueeze(b_5[:,0,:,:],1), torch.unsqueeze(b_6[:,0,:,:],1), torch.unsqueeze(b_7[:,0,:,:],1), torch.unsqueeze(b_8[:,0,:,:],1)], dim=1))
        b_final_1 = self.combine_b_1(torch.cat([torch.unsqueeze(b_5[:,1,:,:],1), torch.unsqueeze(b_6[:,1,:,:],1), torch.unsqueeze(b_7[:,1,:,:],1), torch.unsqueeze(b_8[:,1,:,:],1)], dim=1))
        b_final_2 = self.combine_b_2(torch.cat([torch.unsqueeze(b_5[:,2,:,:],1), torch.unsqueeze(b_6[:,2,:,:],1), torch.unsqueeze(b_7[:,2,:,:],1), torch.unsqueeze(b_8[:,2,:,:],1)], dim=1))
        b_final = torch.cat([b_final_0, b_final_1, b_final_2], dim=1)

        return a_final * x_h + b_final


class G_HuaZhu_3(nn.Module):
    """
    Modify G_MM_1 model to increase innovation
    Motivated by HuaZhu Fu
    Author: Shihao Zhang
    Time: 2019/1/14
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_HuaZhu_3, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter_my(radius, eps)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)


        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)

        a_5, b_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
        a_6, b_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
        a_7, b_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
        a_8, b_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))

        a_final = (a_5+a_6+a_7+a_8) / 4
        b_final = (b_5+b_6+b_7+b_8) / 4

        return a_final * x_h + b_final


class G_HuaZhu_4(nn.Module):
    """
    Modify G_MM_1 model to increase innovation
    Motivated by HuaZhu Fu
    Author: Shihao Zhang
    Time: 2019/1/15
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_HuaZhu_4, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.combine1 = nn.Conv2d(12, 3, kernel_size=1, padding=0, stride=1, bias=True)
        self.combine2 = nn.Conv2d(12, 3, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter_my(radius, eps)

        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, cn, 1, bias=False),
        #     AdaptiveNorm(cn),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(cn, 1, 1)
        # )

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)
        # print(side_5.shape)
        # print(x_4.shape)
        # print(self.guided_map(x_4).shape)
        # print('------')

        a_5, b_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
        a_6, b_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
        a_7, b_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
        a_8, b_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))
        a_final = torch.cat([a_5, a_6, a_7, a_8], dim=1)
        b_final = torch.cat([a_5, a_6, a_7, a_8], dim=1)

        a_final = a_final.float()
        b_final = b_final.float()
        a_final = self.combine1(a_final)
        b_final = self.combine1(b_final)

        out = a_final * x_h + b_final
        a_final, b_final = self.gf(self.guided_map(x_h), out, self.guided_map(x_h))

        return a_final * x_h + b_final


class G_Up(nn.Module):
    """
    Modify G_MM_1 model to increase innovation
    Motivated by HuaZhu Fu
    Author: Shihao Zhang
    Time: 2019/1/12
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_Up, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter(radius, eps)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv4, up5,conv3)
        up7 = self.up7(conv3, up6,conv2)
        up8 = self.up8(conv2, up7,conv1)

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)

        side_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
        side_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
        side_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
        side_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4

        ave_out = self.gf(self.guided_map(x_h), ave_out, self.guided_map(x_h))
        return [ave_out, side_5, side_6, side_7, side_8]

# class G_MM_1_new(nn.Module):
#     """
#     Mulit Guided Filter with M-Net
#     Author: Shihao Zhang
#     Time: 2018/10/26
#     """
#
#     def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
#         super(G_MM_1_new, self).__init__()
#
#         # mutli-scale simple convolution
#         self.conv2 = M_Conv(1, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv3 = M_Conv(1, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv4 = M_Conv(1, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#
#         # the down convolution contain concat operation
#         self.down1 = M_Encoder(1, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
#         self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
#         self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
#         self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#
#         # the center
#         self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)
#
#         # the up convolution contain concat operation
#         self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#
#         # the sideoutput
#         self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#         self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#         self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#         self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#         self.gf = FastGuidedFilter(radius, eps)
#
#         self.guided_map = nn.Sequential(
#             nn.Conv2d(3, cn, 1, bias=False),
#             AdaptiveNorm(cn),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(cn, 1, 1),
#             AdaptiveNorm(1)
#         )
#         self.guided_map_h = nn.Sequential(
#             nn.Conv2d(3, cn, 1, bias=False),
#             AdaptiveNorm(cn),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(cn, 1, 1),
#             AdaptiveNorm(1)
#         )
#
#         self.guided_map.apply(weights_init_identity)
#         self.guided_map_h.apply(weights_init_identity)
#
#     def forward(self, x, x_h):
#         _, _, img_shape, _ = x.size()
#         x = self.guided_map(x)
#         x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
#         x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
#         x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
#         conv1, out = self.down1(x)
#         out = torch.cat([self.conv2(x_2), out], dim=1)
#         conv2, out = self.down2(out)
#         out = torch.cat([self.conv3(x_3), out], dim=1)
#         conv3, out = self.down3(out)
#         out = torch.cat([self.conv4(x_4), out], dim=1)
#         conv4, out = self.down4(out)
#         out = self.center(out)
#         up5 = self.up5(conv4, out)
#         up6 = self.up6(conv3, up5)
#         up7 = self.up7(conv2, up6)
#         up8 = self.up8(conv1, up7)
#
#         # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
#         # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
#         # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
#         # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')
#
#         side_5 = self.side_5(up5)
#         side_6 = self.side_6(up6)
#         side_7 = self.side_7(up7)
#         side_8 = self.side_8(up8)
#         # print(side_5.shape)
#         # print(x_4.shape)
#         # print(self.guided_map(x_4).shape)
#         # print('------')
#
#         x_h = self.guided_map_h(x_h)
#
#         side_5 = self.gf(x_4, side_5, x_h)
#         side_6 = self.gf(x_3, side_6, x_h)
#         side_7 = self.gf(x_2, side_7, x_h)
#         side_8 = self.gf(x, side_8, x_h)
#
#         # ave_out = torch.cat([side_5, side_6, side_7, side_8])
#         # ave_out = torch.mean(ave_out, 0)
#         # ave_out = ave_out.unsqueeze(0)
#         ave_out = (side_5 + side_6 + side_7 + side_8) / 4
#
#         ave_out = self.gf(x_h, ave_out, x_h)
#         return [ave_out, side_5, side_6, side_7, side_8]


class G_MM_1_new(nn.Module):
    """
    Mulit Guided Filter with M-Net
    Author: Shihao Zhang
    Time: 2018/10/26
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_MM_1_new, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter(radius, eps)

        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, cn, 1, bias=False),
        #     AdaptiveNorm(cn),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(cn, 1, 1)
        # )

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        self.guided_map_h = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        self.guided_map.apply(weights_init_identity)
        self.guided_map_h.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)
        # print(side_5.shape)
        # print(x_4.shape)
        # print(self.guided_map(x_4).shape)
        # print('------')

        side_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map_h(x_h))
        side_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map_h(x_h))
        side_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map_h(x_h))
        side_8 = self.gf(self.guided_map(x), side_8, self.guided_map_h(x_h))

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4

        ave_out = self.gf(self.guided_map_h(x_h), ave_out, self.guided_map_h(x_h))
        return [ave_out, side_5, side_6, side_7, side_8]



class G_N(nn.Module):
    """
    True trainable Guided Filter
    Author: Shihao Zhang
    Time: 2018/12/29
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_N, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.boxfilter = BoxFilter(radius)
        self.eps = eps
        # self.A = Variable(torch.from_numpy(np.zeros(1024,1024))).long().cuda()
        # self.b = Variable(torch.from_numpy(np.zeros(1024,1024))).long().cuda()
        self.A = Variable(torch.rand(4,n_classes,450,450)).float().cuda()
        self.b = Variable(torch.rand(4,n_classes,450,450)).float().cuda()


        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)


        A_4 = F.upsample(self.A, size=(img_shape , img_shape ), mode='bilinear')
        A_5 = F.upsample(self.A, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        A_6 = F.upsample(self.A, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        A_7 = F.upsample(self.A, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        b_4 = F.upsample(self.b, size=(img_shape , img_shape ), mode='bilinear')
        b_5 = F.upsample(self.b, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        b_6 = F.upsample(self.b, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        b_7 = F.upsample(self.b, size=(img_shape / 8, img_shape / 8), mode='bilinear')

        local_loss = torch.sum(nn.functional.sigmoid(torch.abs(A_4*self.guided_map(x)+b_4 - side_8) + self.eps*torch.sum(A_4*A_4)))
        local_loss = local_loss + torch.sum(nn.functional.sigmoid(torch.abs(A_5*self.guided_map(x_2)+b_5 - side_7) + self.eps*torch.sum(A_5*A_5)))
        local_loss = local_loss + torch.sum(nn.functional.sigmoid(torch.abs(A_6*self.guided_map(x_3)+b_6 - side_6)+ self.eps*torch.sum(A_6*A_6)))
        local_loss = local_loss + torch.sum(nn.functional.sigmoid(torch.abs(A_7*self.guided_map(x_4)+b_7 - side_5) + self.eps*torch.sum(A_7*A_7)))

        ave_out = self.A*self.guided_map(x_h)+self.b
        return [ave_out, local_loss]


class G_MM_1_modified(nn.Module):
    """
    Mulit Guided Filter with M-Net and adding kernel
    Author: Shihao Zhang
    Time: 2018/12/12
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_MM_1_modified, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)

        # the sideoutput
        self.classify_5 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_6 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_7 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_8 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)



        self.gf = FastGuidedFilter(radius, eps)

        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, cn, 1, bias=False),
        #     AdaptiveNorm(cn),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(cn, 1, 1)
        # )

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_5 = self.classify_5(side_5)
        side_6 = self.side_6(up6)
        side_6 = self.classify_6(side_6)
        side_7 = self.side_7(up7)
        side_7 = self.classify_7(side_7)
        side_8 = self.side_8(up8)
        side_8 = self.classify_8(side_8)
        # print(side_5.shape)
        # print(x_4.shape)
        # print(self.guided_map(x_4).shape)
        # print('------')

        side_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
        side_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
        side_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
        side_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4

        ave_out = self.gf(self.guided_map(x_h), ave_out, self.guided_map(x_h))
        return [ave_out, side_5, side_6, side_7, side_8]


class G_MM_1_modified2(nn.Module):
    """
    Mulit Guided Filter with M-Net and adding kernel
    Author: Shihao Zhang
    Time: 2018/12/12
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_MM_1_modified2, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)

        # the sideoutput
        self.classify = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)



        self.gf = FastGuidedFilter(radius, eps)

        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, cn, 1, bias=False),
        #     AdaptiveNorm(cn),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(cn, 1, 1)
        # )

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_5 = self.classify(side_5)
        side_6 = self.side_6(up6)
        side_6 = self.classify(side_6)
        side_7 = self.side_7(up7)
        side_7 = self.classify(side_7)
        side_8 = self.side_8(up8)
        side_8 = self.classify(side_8)
        # print(side_5.shape)
        # print(x_4.shape)
        # print(self.guided_map(x_4).shape)
        # print('------')

        side_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
        side_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
        side_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
        side_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4

        ave_out = self.gf(self.guided_map(x_h), ave_out, self.guided_map(x_h))
        return [ave_out, side_5, side_6, side_7, side_8]


class G_MM_1_modified3(nn.Module):
    """
    Mulit Guided Filter with M-Net
    Author: Shihao Zhang
    Time: 2018/10/26
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_MM_1_modified3, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)

        self.gf = FastGuidedFilter(radius, eps)

        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, cn, 1, bias=False),
        #     AdaptiveNorm(cn),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(cn, 1, 1)
        # )

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1),
            AdaptiveNorm(1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)
        # print(side_5.shape)
        # print(x_4.shape)
        # print(self.guided_map(x_4).shape)
        # print('------')

        side_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
        side_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
        side_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
        side_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4

        ave_out = self.gf(self.guided_map(x_h), ave_out, self.guided_map(x_h))
        return [ave_out, side_5, side_6, side_7, side_8]


class G_MM_2(nn.Module):
    """
    Mulit Guided Filter with M-Net
    Using different Guided Filter guide different side output
    Author: Shihao Zhang
    Time: 2018/10/29
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_MM_2, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter(radius, eps)
        self.gf5 = FastGuidedFilter(radius, eps)
        self.gf6 = FastGuidedFilter(radius, eps)
        self.gf7 = FastGuidedFilter(radius, eps)
        self.gf8 = FastGuidedFilter(radius, eps)
        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)
        # print(side_5.shape)
        # print(x_4.shape)
        # print(self.guided_map(x_4).shape)
        # print('------')

        side_5 = self.gf5(self.guided_map(x_4), side_5, self.guided_map(x_h))
        side_6 = self.gf6(self.guided_map(x_3), side_6, self.guided_map(x_h))
        side_7 = self.gf7(self.guided_map(x_2), side_7, self.guided_map(x_h))
        side_8 = self.gf8(self.guided_map(x), side_8, self.guided_map(x_h))

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4

        ave_out = self.gf(self.guided_map(x_h), ave_out, self.guided_map(x_h))
        return [ave_out, side_5, side_6, side_7, side_8]


class G_MM_3(nn.Module):
    """
    Mulit Guided Filter with M-Net
    Using different Guided Filter guide different side output,
    and do not guide the average output
    Author: Shihao Zhang
    Time: 2018/10/29
    """

    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(G_MM_3, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        # self.gf = FastGuidedFilter(radius, eps)
        self.gf5 = FastGuidedFilter(radius, eps)
        self.gf6 = FastGuidedFilter(radius, eps)
        self.gf7 = FastGuidedFilter(radius, eps)
        self.gf8 = FastGuidedFilter(radius, eps)
        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x, x_h):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        # side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        # side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        # side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        # side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(up5)
        side_6 = self.side_6(up6)
        side_7 = self.side_7(up7)
        side_8 = self.side_8(up8)
        # print(side_5.shape)
        # print(x_4.shape)
        # print(self.guided_map(x_4).shape)
        # print('------')

        side_5 = self.gf5(self.guided_map(x_4), side_5, self.guided_map(x_h))
        side_6 = self.gf6(self.guided_map(x_3), side_6, self.guided_map(x_h))
        side_7 = self.gf7(self.guided_map(x_2), side_7, self.guided_map(x_h))
        side_8 = self.gf8(self.guided_map(x), side_8, self.guided_map(x_h))

        # ave_out = torch.cat([side_5, side_6, side_7, side_8])
        # ave_out = torch.mean(ave_out, 0)
        # ave_out = ave_out.unsqueeze(0)
        ave_out = (side_5 + side_6 + side_7 + side_8) / 4

        # ave_out = self.gf(self.guided_map(x_h), ave_out, self.guided_map(x_h))
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_deconv(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net_deconv, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, deconv=True)
        self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, deconv=True)
        self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, deconv=True)
        self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, deconv=True)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(img_shape / 2, img_shape / 2), mode='bilinear')
        x_3 = F.upsample(x, size=(img_shape / 4, img_shape / 4), mode='bilinear')
        x_4 = F.upsample(x, size=(img_shape / 8, img_shape / 8), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)
        up5 = self.up5(conv4, out)
        up6 = self.up6(conv3, up5)
        up7 = self.up7(conv2, up6)
        up8 = self.up8(conv1, up7)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = torch.cat([side_5, side_6, side_7, side_8])
        ave_out = torch.mean(ave_out, 0)
        ave_out = ave_out.unsqueeze(0)
        return [ave_out, side_5, side_6, side_7, side_8]


class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv2 = nn.Sequential(
            # conv2
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv3 = nn.Sequential(
            # conv3
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv4 = nn.Sequential(
            # conv4
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv5 = nn.Sequential(
            # conv5
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)
        self.fuse = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        # print x
        _,_,h,w = x.size()

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        ## side output
        d1 = self.dsn1(conv1)
        d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h, w))
        d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h, w))
        d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h, w))
        d5 = F.upsample_bilinear(self.dsn5(conv5), size=(h, w))

        # dsn fusion output
        fuse = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))

        d1 = F.sigmoid(d1)
        d2 = F.sigmoid(d2)
        d3 = F.sigmoid(d3)
        d4 = F.sigmoid(d4)
        d5 = F.sigmoid(d5)
        fuse = F.sigmoid(fuse)

        return [fuse, d1, d2, d3, d4, d5]

# 512*512
class VGG(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super( VGG, self).__init__()

        # 512
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # self.pretrain_model = vgg16(pretrained=True) #(1L, 512L, 16L, 16L)

        self.center = nn.Sequential(
            ConvBnRelu2d(512, 512, kernel_size=3, padding=1, stride=1 ),
        )

        self.up5 = StackDecoder(512, 512, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.up2 = StackDecoder(128, 128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.up1 = StackDecoder(64, 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.classify = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x
        down1 = self.conv_1(out)
        out = F.max_pool2d(down1, kernel_size=2, stride=2)
        down2 = self.conv_2(out)
        out = F.max_pool2d(down2, kernel_size=2, stride=2)
        down3 = self.conv_3(out)
        out = F.max_pool2d(down3, kernel_size=2, stride=2)
        down4 = self.conv_4(out)
        out = F.max_pool2d(down4, kernel_size=2, stride=2)
        down5 = self.conv_5(out)
        out = F.max_pool2d(down5, kernel_size=2, stride=2)

        out = self.center(out)

        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return [out]

class BNM(nn.Module):
    # Boundry Neural Model
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super( BNM, self).__init__()

        #1024
        self.down1 = StackEncoder(  3,   12, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
        self.down2 = StackEncoder( 12,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
        self.down3 = StackEncoder( 24,  36, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down4 = StackEncoder(36,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down5 = StackEncoder(64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        # self.down6 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(128, 128, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        # self.up6 = StackDecoder(256,  256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 128, 128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 64, 64, 36, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 36, 36,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  24,  24,  12, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        self.up1 = StackDecoder(  12,  12,  12, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512

        self.weights = Variable(torch.randn([1024,1024])).cuda()
        self.classify = nn.Conv2d(12, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.boundry_classify = nn.Sequential(
            nn.Conv2d(668, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
        # self.boundry_classify = nn.Sequential(
        #     nn.Conv2d(1052, 1, kernel_size=1, padding=0, stride=1, bias=True),
        #     nn.Sigmoid()
        # )


    def forward(self, x):
        BNM_out = []
        _,_,img_size,_ = x.size()
        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        # down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        BNM_out.append(down1)
        BNM_out.append(down2)
        BNM_out.append(down3)
        BNM_out.append(down4)
        BNM_out.append(down5)
        # BNM_out.append(down6)


        out = self.center(out)
        BNM_out.append(out)

        # out = self.up6(down6, out)
        BNM_out.append(out)
        out = self.up5(down5, out)
        BNM_out.append(out)
        out = self.up4(down4, out)
        BNM_out.append(out)
        out = self.up3(down3, out)
        BNM_out.append(out)
        out = self.up2(down2, out)
        BNM_out.append(out)
        out = self.up1(down1, out)
        BNM_out.append(out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        new_out = []
        for idx, every_out in enumerate(BNM_out):
            tmp = F.upsample(every_out, size=(img_size, img_size), mode='bilinear')
            new_out.append(tmp)
        new_out = torch.cat(new_out,1)
        new_out = new_out*self.weights
        # print new_out.size()
        new_out = self.boundry_classify(new_out)
        return [out, new_out]

class BNM_1 (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(BNM_1, self).__init__()

        #1024
        self.down1 = StackEncoder(  3,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
        self.down2 = StackEncoder( 24,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down6 = StackEncoder(512,  768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768,  768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        self.up1 = StackDecoder(  24,  24,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512
        # self.weights = Variable(torch.randn([1024,1024])).cuda()
        self.classify = nn.Sequential(
            nn.Conv2d(24, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid(),
        )


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)

        # out = nn.Sigmoid()(self.weights*out)
        #1024

        out = self.classify(out)
        # print out.size()
        # out = torch.squeeze(out, dim=1)
        return [out]


class BNM_2(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(BNM_2, self).__init__()

        # 1024
        self.down1 = StackEncoder(3, 24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = StackEncoder(24, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = StackEncoder(64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = StackEncoder(128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.down5 = StackEncoder(256, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.down6 = StackEncoder(512, 768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768, 768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.weights = Variable(torch.randn([1024, 1024])).cuda()
        self.classify = nn.Sequential(
            nn.Conv2d(24, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = x  # ;print('x    ',x.size())
        #
        down1, out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2, out = self.down2(out)  # ;print('down2',down2.size())  #128
        down3, out = self.down3(out)  # ;print('down3',down3.size())  #64
        down4, out = self.down4(out)  # ;print('down4',down4.size())  #32
        down5, out = self.down5(out)  # ;print('down5',down5.size())  #16
        down6, out = self.down6(out)  # ;print('down6',down6.size())  #8
        pass  # ;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)

        out = nn.Sigmoid()(self.weights * out)
        # 1024

        out = self.classify(out)
        # out = torch.squeeze(out, dim=1)
        return [out]

class BNM_3(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(BNM_3, self).__init__()

        # 1024
        self.down1 = StackEncoder(3, 24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = StackEncoder(24, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = StackEncoder(64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = StackEncoder(128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.down5 = StackEncoder(256, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        # self.down6 = StackEncoder(512, 768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(512, 512, kernel_size=3, padding=1, stride=1),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        # self.up6 = StackDecoder(768, 768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512

        self.weights = Variable(torch.randn([1024,1024])).cuda()
        self.classify = nn.Sequential(
            nn.Conv2d(24, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid(),
        )
        self.conv = nn.Conv2d(24, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_1 = nn.Conv2d(24, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_2 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_3 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_4 = nn.Conv2d(256, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_5 = nn.Conv2d(512, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        # self.conv_6 = nn.Conv2d(768, n_classes, kernel_size=3, padding=1, stride=1, bias=True)

    def forward(self, x):
        out = x  # ;print('x    ',x.size())
        _,_,img_shape,_ = x.size()
        #
        down1, out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2, out = self.down2(out)  # ;print('down2',down2.size())  #128
        down3, out = self.down3(out)  # ;print('down3',down3.size())  #64
        down4, out = self.down4(out)  # ;print('down4',down4.size())  #32
        down5, out = self.down5(out)  # ;print('down5',down5.size())  #16
        # down6, out = self.down6(out)  # ;print('down6',down6.size())  #8
        pass  # ;print('out  ',out.size())

        out_5 = self.center(out)
        # out_5 = self.up6(down6, out_6)
        out_4 = self.up5(down5, out_5)
        out_3 = self.up4(down4, out_4)
        out_2 = self.up3(down3, out_3)
        out_1 = self.up2(down2, out_2)
        out = self.up1(down1, out_1)

        out_1 = F.upsample(out_1, size=(img_shape, img_shape), mode='bilinear')
        out_2 = F.upsample(out_2, size=(img_shape, img_shape), mode='bilinear')
        out_3 = F.upsample(out_3, size=(img_shape, img_shape), mode='bilinear')
        out_4 = F.upsample(out_4, size=(img_shape, img_shape), mode='bilinear')
        out_5 = F.upsample(out_5, size=(img_shape, img_shape), mode='bilinear')

        out = self.conv(out)

        out_1 = self.conv_1(out_1)

        out_2 = self.conv_2(out_2)

        out_3 = self.conv_3(out_3)

        out_4 = self.conv_4(out_4)

        out_5 = self.conv_5(out_5)

        out = torch.cat([out, out_1, out_2, out_3, out_4, out_5],dim=1)
        out = self.weights*out
        out = nn.Sigmoid()(out)
        out = self.classify(out)

        return [out]

class UNet1024 (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(UNet1024, self).__init__()

        #1024
        self.down1 = StackEncoder(  3,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
        self.down2 = StackEncoder( 24,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down6 = StackEncoder(512,  768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768,  768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        self.up1 = StackDecoder(  24,  24,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512
        self.classify = nn.Conv2d(24, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return [out]


class UNet1024_kernel (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(UNet1024_kernel, self).__init__()

        #1024
        self.down1 = StackEncoder(  3,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
        self.down2 = StackEncoder( 24,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down6 = StackEncoder(512,  768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768,  768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        self.up1 = StackDecoder(  24,  24,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512
        self.classify = nn.Conv2d(24, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.classify1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = self.classify1(out)
        out = torch.squeeze(out, dim=1)
        return [out]


class Multi_Model (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(Multi_Model, self).__init__()

        #1024
        self.input_1 = StackEncoder(3, 3, kernel_size=7)  # 512
        self.input_2 = StackEncoder(3, 3, kernel_size=7)  # 512
        self.input_3 = StackEncoder(3, 3, kernel_size=7)  # 512

        self.down1 = StackEncoder(  9,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
        self.down2 = StackEncoder( 24,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down6 = StackEncoder(512,  768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768,  768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        self.up1 = StackDecoder(  24,  24,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512
        self.classify = nn.Conv2d(24, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):
        out_1,out_2,out_3 = x                       #;print('x    ',x.size())
        _, out_1 = self.input_1(out_1)
        _, out_2 = self.input_2(out_2)
        _, out_3 = self.input_3(out_3)
        out = torch.cat((out_1, out_2, out_3), dim=1)

                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return [out]

class Small (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(Small, self).__init__()

        #1024
        self.down1 = StackEncoder(  3,   32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   #512
        self.down2 = StackEncoder( 32,   128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   #256
        self.down3 = StackEncoder( 128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   #128
        self.down4 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   # 64
        # self.down5 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   # 32
        # self.down6 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(512, 512, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        # self.up6 = StackDecoder(256,  256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 16
        # self.up5 = StackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 16
        self.up4 = StackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 16
        self.up3 = StackDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 32
        self.up2 = StackDecoder(128, 128, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 32
        self.up1 = StackDecoder(32, 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 256
        self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        # down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        # down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        # pass                          #;print('out  ',out.size())

        out = self.center(out)
        # out = self.up6(down6, out)
        # out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return [out]

class MobileNet (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(MobileNet, self).__init__()

        #1024
        self.down1 = MobileNetEncoder(  3,   32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   #512
        self.down2 = MobileNetEncoder( 32,   128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   #256
        self.down3 = MobileNetEncoder( 128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   #128
        self.down4 = MobileNetEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   # 64
        # self.down5 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   # 32
        # self.down6 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   # 16

        self.center = nn.Sequential(
            MobileNet_block(512, 512, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        # self.up6 = StackDecoder(256,  256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 16
        # self.up5 = StackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 16
        self.up4 = MobileNetDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 16
        self.up3 = MobileNetDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 32
        self.up2 = MobileNetDecoder(128, 128, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 32
        self.up1 = MobileNetDecoder(32, 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 256
        self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        # down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        # down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        # pass                          #;print('out  ',out.size())

        out = self.center(out)
        # out = self.up6(down6, out)
        # out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return [out]


class Patch_Model (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(Patch_Model, self).__init__()

        #1024
        self.down1 = StackEncoder(  3,   12, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
        self.down2 = StackEncoder( 12,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
        self.down3 = StackEncoder( 24,  36, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down4 = StackEncoder(36,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down5 = StackEncoder(64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down6 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(256, 256, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(256,  256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 128, 128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 64, 64, 36, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 36, 36,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  24,  24,  12, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        self.up1 = StackDecoder(  12,  12,  12, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512
        self.classify = nn.Conv2d(12, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        # pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return [out]


# 512x512
class UNet512 (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(UNet512, self).__init__()

        #1024
        self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #64
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #32
        self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down2,out = self.down2(out)   #;print('down2',down2.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())
        down4,out = self.down4(out)   #;print('down4',down4.size())
        down5,out = self.down5(out)   #;print('down5',down5.size())
        down6,out = self.down6(out)   #;print('down6',down6.size())
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return [out]


class UNet512_kernel (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(UNet512_kernel, self).__init__()

        #1024
        self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #64
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #32
        self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        self.classify = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.classify1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down2,out = self.down2(out)   #;print('down2',down2.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())
        down4,out = self.down4(out)   #;print('down4',down4.size())
        down5,out = self.down5(out)   #;print('down5',down5.size())
        down6,out = self.down6(out)   #;print('down6',down6.size())
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = self.classify1(out)
        out = torch.squeeze(out, dim=1)
        return [out]


class GroupNorm (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(GroupNorm, self).__init__()

        #1024
        self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)    #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)    #128
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)    #64
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)    #32
        self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)    #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn) ,
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
        self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down2,out = self.down2(out)   #;print('down2',down2.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())
        down4,out = self.down4(out)   #;print('down4',down4.size())
        down5,out = self.down5(out)   #;print('down5',down5.size())
        down6,out = self.down6(out)   #;print('down6',down6.size())
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return [out]


# 256x256
class UNet256 (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(UNet256, self).__init__()

        #256
        self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8

        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        # self.classify = nn.Sequential(
        #     nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True),
        #     nn.Sigmoid(),
        # )
        self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return [out]

class UNet256_kernel (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(UNet256_kernel, self).__init__()

        #256
        self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8

        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        # self.classify = nn.Sequential(
        #     nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True),
        #     nn.Sigmoid(),
        # )
        self.classify = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.classify1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = self.classify1(out)
        out = torch.squeeze(out, dim=1)
        return [out]


class UNet256_kernel_dgf (nn.Module):
    """
    input should contain x_l and x_h when using guided filter,which is
    different from other models
    """
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(UNet256_kernel_dgf, self).__init__()

        #256
        self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8

        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        # self.classify = nn.Sequential(
        #     nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True),
        #     nn.Sigmoid(),
        # )
        self.classify = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.classify1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        #guided filter
        self.gf = FastGuidedFilter(radius, eps)
        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1)
        )
        self.guided_map.apply(weights_init_identity)


    def forward(self, x,x_h):

        out = x                       #;print('x    ',x.size())
                                      #
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = self.classify1(out)
        out = torch.squeeze(out, dim=1)

        out = self.gf(self.guided_map(x), out, self.guided_map(x_h))

        return [out]


class UNet256_kernel_label (nn.Module):
    """
    This model aims to verify the feasibility of Guided Map
    input should contain x_l and x_h when using guided filter,which is
    different from other models
    Author: Shihao Zhang
    """
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(UNet256_kernel_label, self).__init__()

        #256
        self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8

        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        # self.classify = nn.Sequential(
        #     nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True),
        #     nn.Sigmoid(),
        # )
        self.classify = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.classify1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        #guided filter
        self.gf = FastGuidedFilter(radius, eps)
        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, cn, 1, bias=False),
        #     AdaptiveNorm(cn),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(cn, 1, 1)
        # )
        # self.guided_map.apply(weights_init_identity)


    def forward(self, x,l_h,l_l):

        out = x                       #;print('x    ',x.size())
                                      #
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = self.classify1(out)
        out = torch.squeeze(out, dim=1)


        l_h=torch.unsqueeze(l_h,1)
        l_l = torch.unsqueeze(l_l, 1)
        l_h = l_h.float()
        l_l = l_l.float()


        out = self.gf(l_l, out, l_h)

        return [out]


class UNet256_kernel_figure (nn.Module):
    """
    This model aims to verify the feasibility of Guided Map, and this model do not contain guided map
    Input should contain x_l and x_h when using guided filter,which is
    different from other models
    Author: Shihao Zhang
    Data: 2018/10/23
    """
    def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
        super(UNet256_kernel_figure, self).__init__()

        #256
        self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8

        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        # self.classify = nn.Sequential(
        #     nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True),
        #     nn.Sigmoid(),
        # )
        self.classify = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.classify1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        #guided filter
        self.gf = FastGuidedFilter(radius, eps)
        # self.guided_map = nn.Sequential(
        #     nn.Conv2d(3, cn, 1, bias=False),
        #     AdaptiveNorm(cn),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(cn, 1, 1)
        # )
        # self.guided_map.apply(weights_init_identity)


    def forward(self, x,f_l,f_h):

        out = x                       #;print('x    ',x.size())
                                      #
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = self.classify1(out)
        out = torch.squeeze(out, dim=1)

        f_h=torch.unsqueeze(f_h,1)
        f_l = torch.unsqueeze(f_l, 1)

        # f_h = f_h.float()
        # f_l = f_l.float()

        out = self.gf(f_l, out, f_h)

        return [out]


# 128x128
class UNet128 (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(UNet128, self).__init__()

        #128
        self.down3 = StackEncoder( 3,   128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.classify = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())6

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.classify(out)
        # print out.size()
        out = torch.squeeze(out, dim=1)
        # print out.size()
        return [out]

class UNet512_SideOutput(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(UNet512_SideOutput, self).__init__()

        #1024
        self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #64
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #32
        self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        self.classify_1 = nn.Conv2d(32, n_classes , kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_2 = nn.Conv2d(64, n_classes , kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_3 = nn.Conv2d(128, n_classes , kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_4 = nn.Conv2d(256, n_classes , kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_5 = nn.Conv2d(512, n_classes , kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_6 = nn.Conv2d(1024, n_classes , kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down2,out = self.down2(out)   #;print('down2',down2.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())
        down4,out = self.down4(out)   #;print('down4',down4.size())
        down5,out = self.down5(out)   #;print('down5',down5.size())
        down6,out = self.down6(out)   #;print('down6',down6.size())
        pass                          #;print('out  ',out.size())

        out_6 = self.center(out)
        out_5 = self.up6(down6, out_6)
        out_4 = self.up5(down5, out_5)
        out_3 = self.up4(down4, out_4)
        out_2 = self.up3(down3, out_3)
        out_1 = self.up2(down2, out_2)

        out_1 = self.classify_1(out_1)
        out_1 = torch.squeeze(out_1, dim=1)

        out_2 = self.classify_2(out_2)
        out_2 = torch.squeeze(out_2, dim=1)

        out_3 = self.classify_3(out_3)
        out_3 = torch.squeeze(out_3, dim=1)

        out_4 = self.classify_4(out_4)
        out_4 = torch.squeeze(out_4, dim=1)

        out_5 = self.classify_5(out_5)
        out_5 = torch.squeeze(out_5, dim=1)

        out_6 = self.classify_6(out_6)
        out_6 = torch.squeeze(out_6, dim=1)
        return [out_1,out_2,out_3,out_4,out_5,out_6]

# 1024x1024
class UNet1024_SideOutput(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(UNet1024_SideOutput, self).__init__()

        #1024
        self.down1 = StackEncoder(  3,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
        self.down2 = StackEncoder( 24,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down6 = StackEncoder(512,  768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 , is_bn=bn),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768,  768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        self.up1 = StackDecoder(  24,  24,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512
        self.classify = nn.Conv2d(24, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_1 = nn.Conv2d(24, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_2 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_3 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_4 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_5 = nn.Conv2d(512, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_6 = nn.Conv2d(768, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        self.conv_1 = nn.Conv2d(24, 24, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_5 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_6 = nn.Conv2d(768, 768, kernel_size=3, padding=1, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out_6 = self.center(out)
        out_5 = self.up6(down6, out_6)
        out_4 = self.up5(down5, out_5)
        out_3 = self.up4(down4, out_4)
        out_2 = self.up3(down3, out_3)
        out_1 = self.up2(down2, out_2)
        out = self.up1(down1, out_1)

        # out = self.conv_1(out)
        out = self.classify_1(out)
        out = torch.squeeze(out, dim=1)

        out_1 = self.conv_1(out_1)
        out_1 = self.classify_1(out_1)
        out_1 = torch.squeeze(out_1, dim=1)

        out_2 = self.conv_2(out_2)
        out_2 = self.classify_2(out_2)
        out_2 = torch.squeeze(out_2, dim=1)

        out_3 = self.conv_3(out_3)
        out_3 = self.classify_3(out_3)
        out_3 = torch.squeeze(out_3, dim=1)

        out_4 = self.conv_4(out_4)
        out_4 = self.classify_4(out_4)
        out_4 = torch.squeeze(out_4, dim=1)

        out_5 = self.conv_5(out_5)
        out_5 = self.classify_5(out_5)
        out_5 = torch.squeeze(out_5, dim=1)

        # out_6 = self.classify_6(out_6)
        # out_6 = torch.squeeze(out_6, dim=1)
        return [out, out_1, out_2, out_3, out_4, out_5]



# class resnet_50(nn.Module):
#     def __init__(self, n_classes, pretrain=False, img_size=512):
#         super(resnet_50, self).__init__()
#         self.resnet = resnet50(pretrained=pretrain)
#         self.center = nn.Sequential(
#             ConvBnRelu2d(2048, 2048, kernel_size=3, padding=1, stride=1),
#         )
#         self.up4 = ResStackDecoder(2048, 2048, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up3 = ResStackDecoder(1024, 1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up2 = ResStackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up1 = ResStackDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
#         self.classify = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#         self.img_size = img_size
#
#     def forward(self, x):
#         layer1, layer2, layer3, layer4 = self.resnet(x)
#         center = self.center(layer4)
#         out_4 = self.up4(layer4, center)
#         out_3 = self.up3(layer3, out_4)
#         out_2 = self.up2(layer2, out_3)
#         out_1 = self.up1(layer1, out_2)
#         out = F.upsample(out_1, size=(self.img_size,self.img_size), mode='bilinear')
#         out = self.classify(out)
#         return [out]
#
# class resnet_dense(nn.Module):
#     def __init__(self, n_classes, pretrain=False, img_size=512):
#         super(resnet_dense, self).__init__()
#         self.resnet = resnet50(pretrained=pretrain)
#         self.center = nn.Sequential(
#             ConvBnRelu2d(2048, 2048, kernel_size=3, padding=1, stride=1),
#         )
#         self.up4 = ResStackDecoder(2048, 2048, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up3 = ResStackDecoder(1024, 1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up2 = ResStackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up1 = ResStackDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
#         self.classify = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify1 = nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify2 = nn.Conv2d(512, 128, kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify3 = nn.Conv2d(1024, 128, kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify4 = nn.Conv2d(2048, 128, kernel_size=1, padding=0, stride=1, bias=True)
#         self.img_size = img_size
#
#     def forward(self, x):
#         layer1, layer2, layer3, layer4 = self.resnet(x)
#         center = self.center(layer4)
#         out_4 = self.up4(layer4, center)
#         out_3 = self.up3(layer3, out_4)
#         out_2 = self.up2(layer2, out_3)
#         out_1 = self.up1(layer1, out_2)
#         layer1 = self.classify1(layer1)
#         layer2 = self.classify2(layer2)
#         layer3 = self.classify3(layer3)
#         layer4 = self.classify4(layer4)
#
#         layer1 = F.upsample(layer1, size=(self.img_size, self.img_size), mode='bilinear')
#         layer2 = F.upsample(layer2, size=(self.img_size, self.img_size), mode='bilinear')
#         layer3 = F.upsample(layer3, size=(self.img_size, self.img_size), mode='bilinear')
#         layer4 = F.upsample(layer4, size=(self.img_size, self.img_size), mode='bilinear')
#         out = F.upsample(out_1, size=(self.img_size,self.img_size), mode='bilinear')
#         out = (out + layer1 + layer2 + layer3 + layer4)/5
#         out = self.classify(out)
#         return [out]



class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.UpsamplingNearest2d(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

# 1024x1024
class UNet1024_deconv(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(UNet1024_deconv, self).__init__()

        #1024
        self.down1 = StackEncoder(  3,   24, kernel_size=7, dilation=2)   #512
        self.down2 = StackEncoder( 24,   64, kernel_size=3, dilation=2)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3, dilation=2)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3, dilation=2)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3, dilation=2)   # 32
        self.down6 = StackEncoder(512,  768, kernel_size=3, dilation=2)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 , is_bn=bn),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = Decoder(768,  768, 512, kernel_size=3, dilation=2)  # 16
        self.up5 = Decoder( 512, 512, 256, kernel_size=3, dilation=2)  # 32
        self.up4 = Decoder( 256, 256, 128, kernel_size=3, dilation=2)  # 64
        self.up3 = Decoder( 128, 128,  64, kernel_size=3, dilation=2)  #128
        self.up2 = Decoder(  64,  64,  24, kernel_size=3, dilation=2)  #256
        self.up1 = Decoder(  24,  24,  24, kernel_size=3, dilation=2)  #512
        self.classify = nn.Conv2d(24, n_classes, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return [out]


# 128x128
class UNet128_deconv (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(UNet128_deconv, self).__init__()

        #128
        self.down3 = StackEncoder( 3,   128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = UpsampleConvLayer(1024, 512, kernel_size=3, stride=2, upsample=2)
        self.up5 = UpsampleConvLayer(512, 256, kernel_size=3, stride=2, upsample=2)
        self.up4 = UpsampleConvLayer(256, 128, kernel_size=3, stride=2, upsample=2)
        self.up3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=2, upsample=2)
        self.classify = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self._upsample_add(self.up6(out), down5)
        out = self._upsample_add(self.up5(out), down4)
        out = self._upsample_add(self.up4(out), down3)
        out = self.up3(out)
        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return [out]

# 1024x1024
class FPN_deconv (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(FPN_deconv , self).__init__()

        #1024
        self.down1 = StackEncoder(  3,   24, kernel_size=7, dilation=2)   #512
        self.down2 = StackEncoder( 24,   64, kernel_size=3, dilation=2)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3, dilation=2)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3, dilation=2)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3, dilation=2)   # 32
        self.down6 = StackEncoder(512,  768, kernel_size=3, dilation=2)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 , is_bn=bn),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768,  768, 512, kernel_size=3, dilation=2)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, dilation=2)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, dilation=2)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, dilation=2)  #128
        self.up2 = StackDecoder(  64,  64,  24, kernel_size=3, dilation=2)  #256
        self.up1 = StackDecoder(  24,  24,  24, kernel_size=3, dilation=2)  #512
        self.classify = nn.Conv2d(24, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def _mutil_ave_pooling(self, x, kernel_size=[2,3,4]):
        output = x
        _, _, H, W = x.size()

        for k in kernel_size:
            avg_P = F.avg_pool2d(x, kernel_size=k, stride=k)
            output += F.upsample(avg_P, size=(H, W), mode='bilinear')
        return output/len(kernel_size)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        # out = self._mutil_ave_pooling(out, kernel_size=[2,3,6])
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return [out]


class BCELoss2d(nn.Module):
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        #self.bce_loss = StableBCELoss()
    def forward(self, logits, labels):
        logits_flat = logits.view (-1)
        labels_flat = labels.view(-1)
        return self.bce_loss(logits_flat, labels_flat)

# if __name__ == '__main__':
#
#     CARVANA_HEIGHT = 1280
#     CARVANA_WIDTH  = 1918
#     batch_size  = 1
#     C,H,W = 3,512,512    #3,CARVANA_HEIGHT,CARVANA_WIDTH
#
#     num_classes = 4
#
#     inputs = torch.randn(batch_size,C,H,W)
#     labels = torch.FloatTensor(batch_size,H,W).random_(4).type(torch.LongTensor)
#     lossfunc = nn.NLLLoss2d()
#
#     for model in [UNet1024_deconv,UNet128_deconv,UNet128, UNet256, UNet512, UNet1024, UNet512_SideOutput, UNet1024_SideOutput, resnet_50, resnet_dense]:
#         net = model(n_classes=num_classes).train()
#         x = Variable(inputs)
#         y = Variable(labels)
#         # net = model(n_classes=num_classes).cuda().train()
#         # x = Variable(inputs.cuda())
#         # y = Variable(labels.cuda())
#         logits = net.forward(x)
#         output_size = len(logits)
#         if output_size==1:
#
#             loss = lossfunc(logits[0], y)
#             print logits[0].size()
#             print loss.data[0]
#             loss.backward()
#         else:
#             print 'Side_Output'


# ============================================
# guided filter
# ----------------------------------------------


def weights_init_identity(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n_out, n_in, h, w = m.weight.data.size()
        # Last Layer
        if n_out < n_in:
            init.xavier_uniform(m.weight.data)
            return

        # Except Last Layer
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0

    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 1.0)
        init.constant(m.bias.data,   0.0)


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-4):
        super(DeepGuidedFilter, self).__init__()

        self.gf = FastGuidedFilter(radius, eps)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, 15, 1, bias=False),
            AdaptiveNorm(15),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(15, 1, 1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x_lr, y_lr, x_hr):
        return self.gf(self.guided_map(x_lr), y_lr, self.guided_map(x_hr))
       # return self.gf(self.guided_map(x_lr), y_lr, self.guided_map(x_hr)).clamp(0, 1)

class UGF(nn.Module):
    """
    input should contain x_l and x_h when using guided filter,which is
    different from other models
    """
    def __init__(self, n_classes, radius=5, eps=1e-1, cn=15, bn=False, BatchNorm=False):
        super(UGF, self).__init__()

        #256
        self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8

        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
        # self.classify = nn.Sequential(
        #     nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True),
        #     nn.Sigmoid(),
        # )
        self.classify = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
        self.classify1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

        #guided filter
        self.gf = FastGuidedFilter(radius, eps)
        self.guided_map = nn.Sequential(
            nn.Conv2d(3, cn, 1, bias=False),
            AdaptiveNorm(cn),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cn, 1, 1)
        )
        self.guided_map.apply(weights_init_identity)


    def forward(self, x,x_h):

        out = x                       #;print('x    ',x.size())
                                      #
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = self.classify1(out)
        out = torch.squeeze(out, dim=1)

        out = self.gf(self.guided_map(x), out, self.guided_map(x_h))

        return [out]


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        # out = self.gamma*out + x
        return out

class GridAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(GridAttentionBlock, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1)

        self.phi = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))
        # sigm_psi_f = self.softmax(self.psi(f))

        ## upsample the attentions and multiply
        # sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='bilinear')
        # y = sigm_psi_f.expand_as(x) * x
        # W_y = self.W(y)

        return sigm_psi_f


class GridAttentionBlock_2(nn.Module):
    def __init__(self, in_channels,gating_channels = 512):
        super(GridAttentionBlock_2, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = gating_channels

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1)

        self.phi = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )


    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        ## upsample the attentions and multiply
        # sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='bilinear')
        # y = sigm_psi_f.expand_as(x) * x
        # W_y = self.W(y)

        return sigm_psi_f



"""
loss function
"""


class Weight_MSE_My(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(Weight_MSE_My, self).__init__()
        self.eps = 1e-6
        self.softmax = nn.Softmax()

    def forward(self, X, Y, W):
        # W[W==1]=0
        W[W==0]=1
        W[W==3]=0
        W= W.float()
        W = W/(torch.sum(W) + self.eps)
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        error = error * W
        loss = torch.sum(error)
        return loss
