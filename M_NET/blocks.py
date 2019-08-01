# unet from scratch
# from common import *
# from net.segmentation.loss import *

import torch
import torch.nn as nn
import torch.nn.functional as F

# baseline 128x128, 256x256, 512x512 for experiments -----------------------------------------------

BN_EPS = 1e-4  #1e-4  #1e-5



def merge_bn_in_net(net):
    print ('merging bn ....')
    for m in net.modules():
        if isinstance(m, (StackEncoder, StackDecoder)):
            for mm in m.modules():
                if isinstance(mm, (ConvBnRelu2d,)):
                    print('merging ...')
                    mm.merge_bn()

class MobileNet_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, num_groups=32):
        super(MobileNet_block, self).__init__()
        in_groups = out_groups = num_groups
        if in_channels//num_groups == 0:
            in_groups = 1
        if out_channels//num_groups == 0:
            out_groups = 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                      dilation=dilation,
                      groups=groups, bias=False),
            nn.GroupNorm(in_groups, in_channels, eps=BN_EPS),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=dilation,
                      groups=groups, bias=False),
            nn.GroupNorm(out_groups, out_channels, eps=BN_EPS),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class MobileNetEncoder (nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3, dilation=1, bn=False, BatchNorm=False, num_groups=32):
        super(MobileNetEncoder, self).__init__()
        padding=(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            MobileNet_block(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, num_groups=num_groups),
            MobileNet_block(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, num_groups=num_groups),
        )

    def forward(self,x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class MobileNetDecoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, dilation=1, bn=False, BatchNorm=False, num_groups=32):
        super(MobileNetDecoder, self).__init__()
        padding = (dilation*kernel_size-1)//2

        self.decode = nn.Sequential(
            MobileNet_block(x_big_channels+x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, num_groups=num_groups),
            MobileNet_block(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, num_groups=num_groups),
            MobileNet_block(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1,num_groups=num_groups),
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        y = F.upsample(x, size=(H,W),mode='bilinear')
        #y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return y


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=False, BatchNorm=False, is_relu=True, num_groups=32):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        if BatchNorm:
            self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if is_bn:
            if out_channels//num_groups==0:
                num_groups=1
            self.gn  =nn.GroupNorm(num_groups, out_channels, eps=BN_EPS)
        self.is_bn = is_bn
        self.is_BatchNorm=BatchNorm
        if is_relu is False: self.relu=None

    def forward(self,x):
        x = self.conv(x)
        if self.is_BatchNorm: x = self.bn(x)
        if self.is_bn: x = self.gn(x)
        if self.relu is not None: x = self.relu(x)
        return x


    def merge_bn(self):
        if self.bn == None: return

        assert(self.conv.bias==None)
        conv_weight     = self.conv.weight.data
        bn_weight       = self.bn.weight.data
        bn_bias         = self.bn.bias.data
        bn_running_mean = self.bn.running_mean
        bn_running_var  = self.bn.running_var
        bn_eps          = self.bn.eps

        #https://github.com/sanghoon/pva-faster-rcnn/issues/5
        #https://github.com/sanghoon/pva-faster-rcnn/commit/39570aab8c6513f0e76e5ab5dba8dfbf63e9c68c

        N,C,KH,KW = conv_weight.size()
        std = 1/(torch.sqrt(bn_running_var+bn_eps))
        std_bn_weight =(std*bn_weight).repeat(C*KH*KW,1).t().contiguous().view(N,C,KH,KW )
        conv_weight_hat = std_bn_weight*conv_weight
        conv_bias_hat   = (bn_bias - bn_weight*std*bn_running_mean)

        self.bn   = None
        self.conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                              padding=self.conv.padding, stride=self.conv.stride, dilation=self.conv.dilation, groups=self.conv.groups,
                              bias=True)
        self.conv.weight.data = conv_weight_hat #fill in
        self.conv.bias.data   = conv_bias_hat



class ConvResidual (nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvResidual, self).__init__()

        self.block = nn.Sequential(
            ConvBnRelu2d(in_channels,  out_channels, kernel_size=3, padding=1,  stride=1 ),
            ConvBnRelu2d(out_channels, out_channels, kernel_size=3, padding=1,  stride=1, is_relu=False),
        )
        self.shortcut = None
        if in_channels!=out_channels or stride!=1:
           self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride,  bias=True)

    def forward(self, x):
        r = x if self.shortcut is None else self.shortcut(x)
        x = self.block(x)
        x = F.relu(x+r, inplace=True)
        return x



## -----------------------------------------------------------------------------------------------------------

## origainl 3x3 stack filters used in UNet
class StackEncoder (nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3, dilation=1, bn=False, BatchNorm=False, num_groups=32):
        super(StackEncoder, self).__init__()
        padding=(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self,x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class StackDecoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, dilation=1, bn=False, BatchNorm=False, num_groups=32):
        super(StackDecoder, self).__init__()
        padding=(dilation*kernel_size-1)//2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels+x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        y = F.upsample(x, size=(H,W),mode='bilinear')
        #y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return y

class Decoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, dilation=1):
        super(Decoder, self).__init__()
        padding = (dilation*kernel_size-1)//2

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(x_big_channels+x_channels, y_channels, kernel_size, stride=1, padding=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1),
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        y = F.upsample(x, size=(H,W),mode='bilinear')
        #y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return  y

##---------------------------------------------------------------

class M_Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False, BatchNorm=False, num_groups=32):
        super(M_Encoder, self).__init__()
        padding =(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )
        self.pooling = pooling

    def forward(self, x):
        conv = self.encode(x)
        if self.pooling:
            pool = F.max_pool2d(conv, kernel_size=2, stride=2)
            return conv,pool
        else:
            return conv

class M_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False, BatchNorm=False, num_groups=32):
        super(M_Conv, self).__init__()
        padding =(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,kernel_size=kernel_size, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv = self.encode(x)
        return conv

class M_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, deconv=False, bn=False, BatchNorm=False, num_groups=32):
        super(M_Decoder, self).__init__()
        padding =(dilation*kernel_size-1)//2
        if deconv:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=1, padding=1),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                             stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1,is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            )
        else:
            self.deconv = False

        self.decode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        out = F.upsample(x, size=(H,W),mode='bilinear')
        out = torch.cat([x_big,out], dim=1)
        if self.deconv:
            out = self.deconv(out)
        else:
            out = self.decode(out)
        return out


##---------------------------------------------------------------

class BNM_Encoder(nn.Module):
    def __init__(self, layers, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True):
        super(BNM_Encoder, self).__init__()
        padding =(dilation*kernel_size-1)//2

        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pooling = pooling

    def forward(self, x):
        conv = self.encode(x)
        if self.pooling:
            pool = F.max_pool2d(conv, kernel_size=2, stride=2)
            return conv,pool
        else:
            return conv

class BNM_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, deconv = False):
        super(BNM_Decoder, self).__init__()
        padding =(dilation*kernel_size-1)//2
        if deconv:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=1, padding=1),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                             stride=1, groups=1),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1),
            )
        else:
            self.deconv = False

        self.decode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1),
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        out = F.upsample(x, size=(H,W),mode='bilinear')
        out = torch.cat([x_big,out], dim=1)
        if self.deconv:
            out = self.deconv(out)
        else:
            out = self.decode(out)
        return out


## origainl 3x3 stack filters used in UNet
class ResStackEncoder (nn.Module):
    def __init__(self, x_channels, y_channels):
        super(ResStackEncoder, self).__init__()
        self.encode = ConvResidual(x_channels, y_channels)

    def forward(self,x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class ResStackDecoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(ResStackDecoder, self).__init__()
        padding=(kernel_size-1)//2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels+x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            ConvResidual(y_channels, y_channels)
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        y = F.upsample(x, size=(H,W),mode='bilinear')
        #y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return  y



# main #################################################################
# if __name__ == '__main__':
#     print( '%s: calling main function ... ' % os.path.basename(__file__))
#
#     #make_dummy_cbr()
#     #make_dummy_cbr()
#     check_dummy_cbr2()
#
#     print('\nsucess!')