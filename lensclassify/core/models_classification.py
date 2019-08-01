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

from resNet_model import resnet50
from VGG_model import vgg16

from guided_filter_pytorch.guided_filter import GuidedFilter, FastGuidedFilter

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"


# ------------------------------------resnet--------------------------------------------------------------------



# ------------------------------------U-Net--------------------------------------------------------------------
# baseline 128x128, 256x256, 512x512, 1024x1024 for experiments -----------------------------------------------


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

        # guided filter
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


class UNet256_kernel_classification (nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(UNet256_kernel_classification, self).__init__()

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

        self.guided_classification = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            AdaptiveNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 64, 1)
        )
        self.guided_classification.apply(weights_init_identity)

        self.fc1=nn.Linear(64*8*8,1024)
        self.fc2=nn.Linear(1024,2)



    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8


        out = self.center(out)

        out_classification=self.guided_classification(out)
        out_classification = out_classification.view((out.size()[0], -1))
        out_classification = F.relu(self.fc1(out_classification))
        out_classification = F.relu(self.fc2(out_classification))
        out_classification = F.sigmoid(out_classification)
        out_classification = torch.squeeze(out_classification, dim=1)

        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = self.classify1(out)
        out = torch.squeeze(out, dim=1)
        return [out, out_classification]


# ----------------------------------------------
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



