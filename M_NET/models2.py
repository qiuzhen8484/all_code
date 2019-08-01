import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler

# from common import *
# from net.segmentation.loss import *
from blocks2 import *

import torch
import torch.nn as nn
import torch.nn.functional as F

# baseline 128x128, 256x256, 512x512, 1024x1024 for experiments -----------------------------------------------

BN_EPS = 1e-4  #1e-4  #1e-5

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=False, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UNet512 (nn.Module):
    def __init__(self, in_shape):
        super(UNet512, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #1024
        self.down2 = StackEncoder(  C,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   #64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   #32
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.classify = nn.Conv2d(32, 4, kernel_size=1, padding=0, stride=1, bias=True)

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
        return out


class StackEncoderNoBn (nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3):
        super(StackEncoder, self).__init__()
        padding=(kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
        )

    def forward(self,x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small

class UNet512MultiScale (nn.Module):
    def __init__(self, in_shape):
        super(UNet512MultiScale, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #1024
        self.down2 = StackEncoder(  C,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   #64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   #32
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.classify = nn.Conv2d(32, 4, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
        down2,out = self.down2(out)   #;print('down2',down2.size())
        down3,out = self.down3(out)   #;print('down3',down3.size())
        down4,out = self.down4(out)   #;print('down4',down4.size())
        down5,out = self.down5(out)   #;print('down5',down5.size())
        down6,out = self.down6(out)   #;print('down6',down6.size())
                                      #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out

class My_UNet512 (nn.Module):
    def __init__(self, in_shape):
        super(My_UNet512, self).__init__()
        n_class = 4
        C,H,W = in_shape
        #assert(C==3)

        #1024
        self.down2 = StackEncoder(  C,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   #64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   #32
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1),
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.classify_1 = nn.Conv2d(32, n_class , kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_2 = nn.Conv2d(64, n_class , kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_3 = nn.Conv2d(128, n_class , kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_4 = nn.Conv2d(256, n_class , kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_5 = nn.Conv2d(512, n_class , kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_6 = nn.Conv2d(1024, n_class , kernel_size=1, padding=0, stride=1, bias=True)



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
        return out_1,out_2,out_3,out_4,out_5,out_6

# 1024x1024
class My_UNet1024(nn.Module):
    def __init__(self, in_shape):
        super(My_UNet1024, self).__init__()
        C,H,W = in_shape
        n_class = 4
        self.img_size = H
        #assert(C==3)

        #1024
        self.down1 = StackEncoder(  C,   24, kernel_size=3)   #512
        self.down2 = StackEncoder( 24,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 32
        self.down6 = StackEncoder(512,  768, kernel_size=3)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768,  768, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  24, kernel_size=3)  #256
        self.up1 = StackDecoder(  24,  24,  24, kernel_size=3)  #512
        # self.classify = nn.Conv2d(24, 4, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_1 = nn.Conv2d(24, n_class, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_2 = nn.Conv2d(64, n_class, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_3 = nn.Conv2d(128, n_class, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_4 = nn.Conv2d(256, n_class, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_5 = nn.Conv2d(512, n_class, kernel_size=1, padding=0, stride=1, bias=True)
        self.classify_6 = nn.Conv2d(768, n_class, kernel_size=1, padding=0, stride=1, bias=True)


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

        out_1 = F.upsample(out_1, size=(self.img_size, self.img_size), mode='bilinear')
        out_2 = F.upsample(out_2, size=(self.img_size, self.img_size), mode='bilinear')
        out_3 = F.upsample(out_3, size=(self.img_size, self.img_size), mode='bilinear')
        out_4 = F.upsample(out_4, size=(self.img_size, self.img_size), mode='bilinear')
        out_5 = F.upsample(out_5, size=(self.img_size, self.img_size), mode='bilinear')
        out_6 = F.upsample(out_6, size=(self.img_size, self.img_size), mode='bilinear')

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
        return [out_1, out_2, out_3, out_4, out_5, out_6]



# 1024x1024
class UNet1024 (nn.Module):
    def __init__(self, in_shape):
        super(UNet1024, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #1024
        self.down1 = StackEncoder(  C,   24, kernel_size=3)   #512
        self.down2 = StackEncoder( 24,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 32
        self.down6 = StackEncoder(512,  768, kernel_size=3)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768,  768, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  24, kernel_size=3)  #256
        self.up1 = StackDecoder(  24,  24,  24, kernel_size=3)  #512
        self.classify = nn.Conv2d(24, 4, kernel_size=1, padding=0, stride=1, bias=True)


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
        return out


# 256x256
class UNet256 (nn.Module):
    def __init__(self, in_shape):
        super(UNet256, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #256
        self.down2 = StackEncoder(  C,   64, kernel_size=3)   #128
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 16
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #  8

        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.classify = nn.Conv2d(32, 2, kernel_size=1, padding=0, stride=1, bias=True)


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
        return out

class UNetAverageBatchSize (nn.Module):
    def __init__(self, in_shape):
        super(UNet256, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #256
        self.down2 = StackEncoder(  C,   64, kernel_size=3)   #128
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   # 64
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 32
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 16

        self.center = nn.Sequential(
            #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.classify = nn.Conv2d(32, 2, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16

        out = self.center(out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out

#Unet BatchSize 1:
class UNetAverageBatchSize(nn.Module):
    def __init__(self, in_shape):
        super(UNet256, self).__init__()
        C, H, W = in_shape
        # assert(C==3)

        # 256
        self.down2 = StackEncoder(C, 64, kernel_size=3)  # 128
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 64
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
        self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16

        self.center = nn.Sequential(
            # ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1),
        )

        # 8
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 32, kernel_size=3)  # 256
        self.classify = nn.Conv2d(32, 2, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        out = x  # ;print('x    ',x.size())
        #
        down2, out = self.down2(out)  # ;print('down2',down2.size())  #128
        down3, out = self.down3(out)  # ;print('down3',down3.size())  #64
        down4, out = self.down4(out)  # ;print('down4',down4.size())  #32
        down5, out = self.down5(out)  # ;print('down5',down5.size())  #16

        out = self.center(out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out

