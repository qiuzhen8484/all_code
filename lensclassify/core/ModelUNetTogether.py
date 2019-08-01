import torch
import torch.nn as nn
import torch.nn.functional as F
BN_EPS = 1e-4

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True, is_relu=True):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if is_bn   is False: self.bn  =None
        if is_relu is False: self.relu=None

    def forward(self,x):
        x = self.conv(x)
        if self.bn   is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x

class StackEncoder (nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3, Bn = False, Dialation = 1):
        super(StackEncoder, self).__init__()
        padding=(kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=Dialation, stride=1, groups=1, is_bn=Bn),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=Dialation, stride=1, groups=1, is_bn=Bn),
        )

    def forward(self,x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small

class StackDecoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, Bn = False, Dialation = 1):
        super(StackDecoder, self).__init__()
        padding=(kernel_size-1)//2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels+x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=Dialation, stride=1, groups=1, is_bn=Bn),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=Dialation, stride=1, groups=1, is_bn=Bn),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=Dialation, stride=1, groups=1, is_bn=Bn),
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        y = F.upsample(x, size=(H,W),mode='bilinear')
        #y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return  y

class Classifier_Module(nn.Module):

    def __init__(self,dilation_series,padding_series):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(32,4, kernel_size=3,stride=1, padding =padding, dilation = dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class UNet(nn.Module):
    def __init__(self, n_classes=4):
        super(UNet, self).__init__()

        #1024
        self.down2 = StackEncoder(  3,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   #64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   #32
        self.down6 = StackEncoder(512, 1024, kernel_size=3)   #16

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1, is_bn=False),
            #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
        )

        # 16
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(1024,1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  32, kernel_size=3)  #256
        self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.classifyASPP = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24])

    def _make_pred_layer(self,block, dilation_series, padding_series):
        return block(dilation_series,padding_series)

    def MutiScaleForward(self, x):
        input_size = x.size()
        imwidth = input_size[2]
        imheight = input_size[3]

        # self.interp1 = nn.UpsamplingBilinear2d(size = (int(imwidth*0.75),int(imheight*0.75))) #Scale 0.75
        self.interp2 = nn.UpsamplingBilinear2d(size = (int(imwidth*0.5), int(imheight*0.5)))  #Scale 0.5

        outOrigin = self.forward(x)

        self.interp3 = nn.UpsamplingBilinear2d(size=(outOrigin.size()[2],outOrigin.size()[3]))

        # out075Origin = self.forward(self.interp1(x))
        out050Origin = self.forward(self.interp2(x))

        # up075_FeatureMap = self.interp3(out075Origin)
        up050_FeatureMap = self.interp3(out050Origin)

        # temp1 = torch.max(outOrigin,up075_FeatureMap)
        # out4 = torch.max(temp1,up050_FeatureMap)
        # out4 = 0.8*outOrigin + 0.2*up050_FeatureMap

        out=[]
        out.append(outOrigin)
        # out.append(out075Origin)
        out.append(out050Origin)
        # out.append(out4)

        return out

    def ASPPforward(self, x):

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
        out = self.classifyASPP(out)
        return out

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