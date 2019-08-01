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

# from resNet_model import resnet50
# from VGG_model import vgg16

# from guided_filter_pytorch.guided_filter import GuidedFilter, FastGuidedFilter

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# ------------------------------------U-Net--------------------------------------------------------------------
# baseline 128x128, 256x256, 512x512, 1024x1024 for experiments -----------------------------------------------

class M_Net(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(M_Net, self).__init__()

        # multi-scale simple convolution
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
        x_2 = F.upsample(x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
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


class U_Net_with_Multi_Scale(nn.Module):
    def __init__(self, input_channle, n_classes, bn=False, BatchNorm=False):
        super(U_Net_with_Multi_Scale, self).__init__()

        # multi-scale simple convolution
        self.conv2 = M_Conv(input_channle, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(input_channle, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(input_channle, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(input_channle, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
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
        # self.side_5 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        # self.side_6 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        # self.side_7 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
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
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        # side_5 = self.side_5(side_5)
        # side_6 = self.side_6(side_6)
        # side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        # ave_out = (side_5+side_6+side_7+side_8)/4
        return [side_8]


# class GM(nn.Module):
#     """
#     Guided Filter with M-Net
#     Author: Shihao Zhang
#     Time: 2018/10/26
#     """
#     def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
#         super(GM, self).__init__()
#
#         # mutli-scale simple convolution
#         self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#
#         # the down convolution contain concat operation
#         self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
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
#         self.guided_map = nn.Sequential(
#             nn.Conv2d(3, cn, 1, bias=False),
#             AdaptiveNorm(cn),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(cn, 1, 1)
#         )
#         self.guided_map.apply(weights_init_identity)
#
#     def forward(self, x,x_h):
#         _, _, img_shape, _ = x.size()
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
#         side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
#         side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
#         side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
#         side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')
#
#         side_5 = self.side_5(side_5)
#         side_6 = self.side_6(side_6)
#         side_7 = self.side_7(side_7)
#         side_8 = self.side_8(side_8)
#
#         # ave_out = torch.cat([side_5, side_6, side_7, side_8])
#         # ave_out = torch.mean(ave_out, 0)
#         # ave_out = ave_out.unsqueeze(0)
#         ave_out = (side_5+side_6+side_7+side_8)/4
#
#         ave_out = self.gf(self.guided_map(x), ave_out, self.guided_map(x_h))
#         return [ave_out, side_5, side_6, side_7, side_8]
#
#
# class GF(nn.Module):
#     """
#     Guided Filter using figure with M-Net
#     Author: Shihao Zhang
#     Time: 2018/10/26
#     """
#     def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
#         super(GF, self).__init__()
#
#         # mutli-scale simple convolution
#         self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#
#         # the down convolution contain concat operation
#         self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
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
#         # self.guided_map = nn.Sequential(
#         #     nn.Conv2d(3, cn, 1, bias=False),
#         #     AdaptiveNorm(cn),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     nn.Conv2d(cn, 1, 1)
#         # )
#         # self.guided_map.apply(weights_init_identity)
#
#     def forward(self, x,x_l,x_h):
#         _, _, img_shape, _ = x.size()
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
#         side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
#         side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
#         side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
#         side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')
#
#         side_5 = self.side_5(side_5)
#         side_6 = self.side_6(side_6)
#         side_7 = self.side_7(side_7)
#         side_8 = self.side_8(side_8)
#
#         # ave_out = torch.cat([side_5, side_6, side_7, side_8])
#         # ave_out = torch.mean(ave_out, 0)
#         # ave_out = ave_out.unsqueeze(0)
#         ave_out = (side_5+side_6+side_7+side_8)/4
#
#         x_h=torch.unsqueeze(x_h,1)
#         x_l = torch.unsqueeze(x_l, 1)
#
#         ave_out = self.gf(x_l, ave_out, x_h)
#         return [ave_out, side_5, side_6, side_7, side_8]
#
#
# class G_MM(nn.Module):
#     """
#     Multi Guided Filter with M-Net
#     Author: Shihao Zhang
#     Time: 2018/10/26
#     """
#
#     def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
#         super(G_MM, self).__init__()
#
#         # mutli-scale simple convolution
#         self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#
#         # the down convolution contain concat operation
#         self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
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
#         self.guided_map = nn.Sequential(
#             nn.Conv2d(3, cn, 1, bias=False),
#             AdaptiveNorm(cn),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(cn, 1, 1)
#         )
#         self.guided_map.apply(weights_init_identity)
#
#     def forward(self, x, x_h):
#         _, _, img_shape, _ = x.size()
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
#         side_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
#         side_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
#         side_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
#         side_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))
#
#         # ave_out = torch.cat([side_5, side_6, side_7, side_8])
#         # ave_out = torch.mean(ave_out, 0)
#         # ave_out = ave_out.unsqueeze(0)
#         ave_out = (side_5 + side_6 + side_7 + side_8) / 4
#
#         # ave_out = self.gf(self.guided_map(x), ave_out, self.guided_map(x_h))
#         return [ave_out, side_5, side_6, side_7, side_8]
#
#
# class G_MM_1(nn.Module):
#     """
#     Multi Guided Filter with M-Net
#     Author: Shihao Zhang
#     Time: 2018/10/26
#     """
#
#     def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
#         super(G_MM_1, self).__init__()
#
#         # mutli-scale simple convolution
#         self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#
#         # the down convolution contain concat operation
#         self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
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
#         self.guided_map = nn.Sequential(
#             nn.Conv2d(3, cn, 1, bias=False),
#             AdaptiveNorm(cn),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(cn, 1, 1),
#             AdaptiveNorm(1)
#         )
#
#         self.guided_map.apply(weights_init_identity)
#
#     def forward(self, x, x_h):
#         # _, _, img_shape, img_shape2 = x.size()
#         # x_2 = F.upsample(x, size=(img_shape / 2, img_shape2 / 2), mode='bilinear')
#         # x_3 = F.upsample(x, size=(img_shape / 4, img_shape2 / 4), mode='bilinear')
#         # x_4 = F.upsample(x, size=(img_shape / 8, img_shape2 / 8), mode='bilinear')
#         _, _, img_shape, _ = x.size()
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
#         side_5 = self.gf(self.guided_map(x_4), side_5, self.guided_map(x_h))
#         side_6 = self.gf(self.guided_map(x_3), side_6, self.guided_map(x_h))
#         side_7 = self.gf(self.guided_map(x_2), side_7, self.guided_map(x_h))
#         side_8 = self.gf(self.guided_map(x), side_8, self.guided_map(x_h))
#
#         # ave_out = torch.cat([side_5, side_6, side_7, side_8])
#         # ave_out = torch.mean(ave_out, 0)
#         # ave_out = ave_out.unsqueeze(0)
#         ave_out = (side_5 + side_6 + side_7 + side_8) / 4
#
#         ave_out = self.gf(self.guided_map(x_h), ave_out, self.guided_map(x_h))
#         return [ave_out, side_5, side_6, side_7, side_8]
#
#
# class G_MM_2(nn.Module):
#     """
#     Mulit Guided Filter with M-Net
#     Using different Guided Filter guide different side output
#     Author: Shihao Zhang
#     Time: 2018/10/29
#     """
#
#     def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
#         super(G_MM_2, self).__init__()
#
#         # mutli-scale simple convolution
#         self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#
#         # the down convolution contain concat operation
#         self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
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
#         self.gf5 = FastGuidedFilter(radius, eps)
#         self.gf6 = FastGuidedFilter(radius, eps)
#         self.gf7 = FastGuidedFilter(radius, eps)
#         self.gf8 = FastGuidedFilter(radius, eps)
#         self.guided_map = nn.Sequential(
#             nn.Conv2d(3, cn, 1, bias=False),
#             AdaptiveNorm(cn),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(cn, 1, 1)
#         )
#         self.guided_map.apply(weights_init_identity)
#
#     def forward(self, x, x_h):
#         _, _, img_shape, _ = x.size()
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
#         side_5 = self.gf5(self.guided_map(x_4), side_5, self.guided_map(x_h))
#         side_6 = self.gf6(self.guided_map(x_3), side_6, self.guided_map(x_h))
#         side_7 = self.gf7(self.guided_map(x_2), side_7, self.guided_map(x_h))
#         side_8 = self.gf8(self.guided_map(x), side_8, self.guided_map(x_h))
#
#         # ave_out = torch.cat([side_5, side_6, side_7, side_8])
#         # ave_out = torch.mean(ave_out, 0)
#         # ave_out = ave_out.unsqueeze(0)
#         ave_out = (side_5 + side_6 + side_7 + side_8) / 4
#
#         ave_out = self.gf(self.guided_map(x_h), ave_out, self.guided_map(x_h))
#         return [ave_out, side_5, side_6, side_7, side_8]
#
#
# class G_MM_3(nn.Module):
#     """
#     Mulit Guided Filter with M-Net
#     Using different Guided Filter guide different side output,
#     and do not guide the average output
#     Author: Shihao Zhang
#     Time: 2018/10/29
#     """
#
#     def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
#         super(G_MM_3, self).__init__()
#
#         # mutli-scale simple convolution
#         self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#
#         # the down convolution contain concat operation
#         self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
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
#         # self.gf = FastGuidedFilter(radius, eps)
#         self.gf5 = FastGuidedFilter(radius, eps)
#         self.gf6 = FastGuidedFilter(radius, eps)
#         self.gf7 = FastGuidedFilter(radius, eps)
#         self.gf8 = FastGuidedFilter(radius, eps)
#         self.guided_map = nn.Sequential(
#             nn.Conv2d(3, cn, 1, bias=False),
#             AdaptiveNorm(cn),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(cn, 1, 1)
#         )
#         self.guided_map.apply(weights_init_identity)
#
#     def forward(self, x, x_h):
#         _, _, img_shape, _ = x.size()
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
#         side_5 = self.gf5(self.guided_map(x_4), side_5, self.guided_map(x_h))
#         side_6 = self.gf6(self.guided_map(x_3), side_6, self.guided_map(x_h))
#         side_7 = self.gf7(self.guided_map(x_2), side_7, self.guided_map(x_h))
#         side_8 = self.gf8(self.guided_map(x), side_8, self.guided_map(x_h))
#
#         # ave_out = torch.cat([side_5, side_6, side_7, side_8])
#         # ave_out = torch.mean(ave_out, 0)
#         # ave_out = ave_out.unsqueeze(0)
#         ave_out = (side_5 + side_6 + side_7 + side_8) / 4
#
#         # ave_out = self.gf(self.guided_map(x_h), ave_out, self.guided_map(x_h))
#         return [ave_out, side_5, side_6, side_7, side_8]
#
#
# class M_Net_deconv(nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(M_Net_deconv, self).__init__()
#
#         # mutli-scale simple convolution
#         self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#         self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
#
#         # the down convolution contain concat operation
#         self.down1 = M_Encoder(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
#         self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
#         self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
#         self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#
#         # the center
#         self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)
#
#         # the up convolution contain concat operation
#         self.up5 = M_Decoder(512 + 256, 256, kernel_size=3, deconv=True)
#         self.up6 = M_Decoder(256 + 128, 128, kernel_size=3, deconv=True)
#         self.up7 = M_Decoder(128 + 64, 64, kernel_size=3, deconv=True)
#         self.up8 = M_Decoder(64 + 32, 32, kernel_size=3, deconv=True)
#
#         # the sideoutput
#         self.side_5 = nn.Conv2d(256, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.side_6 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.side_7 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.side_8 = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#
#     def forward(self, x):
#         _, _, img_shape, _ = x.size()
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
#         side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
#         side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
#         side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
#         side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')
#
#         side_5 = self.side_5(side_5)
#         side_6 = self.side_6(side_6)
#         side_7 = self.side_7(side_7)
#         side_8 = self.side_8(side_8)
#
#         ave_out = torch.cat([side_5, side_6, side_7, side_8])
#         ave_out = torch.mean(ave_out, 0)
#         ave_out = ave_out.unsqueeze(0)
#         return [ave_out, side_5, side_6, side_7, side_8]
#
#
# class HED(nn.Module):
#     def __init__(self):
#         super(HED, self).__init__()
#         self.conv1 = nn.Sequential(
#             # conv1
#             nn.Conv2d(3, 64, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.ReLU(inplace=True),
#
#         )
#         self.conv2 = nn.Sequential(
#             # conv2
#             nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.ReLU(inplace=True),
#
#         )
#         self.conv3 = nn.Sequential(
#             # conv3
#             nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#
#         )
#         self.conv4 = nn.Sequential(
#             # conv4
#             nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
#             nn.Conv2d(256, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#
#         )
#         self.conv5 = nn.Sequential(
#             # conv5
#             nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#
#         self.dsn1 = nn.Conv2d(64, 1, 1)
#         self.dsn2 = nn.Conv2d(128, 1, 1)
#         self.dsn3 = nn.Conv2d(256, 1, 1)
#         self.dsn4 = nn.Conv2d(512, 1, 1)
#         self.dsn5 = nn.Conv2d(512, 1, 1)
#         self.fuse = nn.Conv2d(5, 1, 1)
#
#     def forward(self, x):
#         # print x
#         _,_,h,w = x.size()
#
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(conv1)
#         conv3 = self.conv3(conv2)
#         conv4 = self.conv4(conv3)
#         conv5 = self.conv5(conv4)
#
#         ## side output
#         d1 = self.dsn1(conv1)
#         d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h, w))
#         d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h, w))
#         d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h, w))
#         d5 = F.upsample_bilinear(self.dsn5(conv5), size=(h, w))
#
#         # dsn fusion output
#         fuse = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))
#
#         d1 = F.sigmoid(d1)
#         d2 = F.sigmoid(d2)
#         d3 = F.sigmoid(d3)
#         d4 = F.sigmoid(d4)
#         d5 = F.sigmoid(d5)
#         fuse = F.sigmoid(fuse)
#
#         return [fuse, d1, d2, d3, d4, d5]
#
# # 512*512
# class VGG(nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super( VGG, self).__init__()
#
#         # 512
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.conv_3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.conv_4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.conv_5 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#         )
#
#         # self.pretrain_model = vgg16(pretrained=True) #(1L, 512L, 16L, 16L)
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(512, 512, kernel_size=3, padding=1, stride=1 ),
#         )
#
#         self.up5 = StackDecoder(512, 512, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
#         self.up2 = StackDecoder(128, 128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
#         self.up1 = StackDecoder(64, 64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
#         self.classify = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#
#     def forward(self, x):
#
#         out = x
#         down1 = self.conv_1(out)
#         out = F.max_pool2d(down1, kernel_size=2, stride=2)
#         down2 = self.conv_2(out)
#         out = F.max_pool2d(down2, kernel_size=2, stride=2)
#         down3 = self.conv_3(out)
#         out = F.max_pool2d(down3, kernel_size=2, stride=2)
#         down4 = self.conv_4(out)
#         out = F.max_pool2d(down4, kernel_size=2, stride=2)
#         down5 = self.conv_5(out)
#         out = F.max_pool2d(down5, kernel_size=2, stride=2)
#
#         out = self.center(out)
#
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#         out = self.up1(down1, out)
#         #1024
#
#         out = self.classify(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
# class BNM(nn.Module):
#     # Boundry Neural Model
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super( BNM, self).__init__()
#
#         #1024
#         self.down1 = StackEncoder(  3,   12, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
#         self.down2 = StackEncoder( 12,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
#         self.down3 = StackEncoder( 24,  36, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down4 = StackEncoder(36,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down5 = StackEncoder(64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         # self.down6 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(128, 128, kernel_size=3, padding=1, stride=1 ),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         # self.up6 = StackDecoder(256,  256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 128, 128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 64, 64, 36, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 36, 36,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  24,  24,  12, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         self.up1 = StackDecoder(  12,  12,  12, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512
#
#         self.weights = Variable(torch.randn([1024,1024])).cuda()
#         self.classify = nn.Conv2d(12, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#         self.boundry_classify = nn.Sequential(
#             nn.Conv2d(668, 1, kernel_size=1, padding=0, stride=1, bias=True),
#             nn.Sigmoid()
#         )
#         # self.boundry_classify = nn.Sequential(
#         #     nn.Conv2d(1052, 1, kernel_size=1, padding=0, stride=1, bias=True),
#         #     nn.Sigmoid()
#         # )
#
#
#     def forward(self, x):
#         BNM_out = []
#         _,_,img_size,_ = x.size()
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         # down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         BNM_out.append(down1)
#         BNM_out.append(down2)
#         BNM_out.append(down3)
#         BNM_out.append(down4)
#         BNM_out.append(down5)
#         # BNM_out.append(down6)
#
#
#         out = self.center(out)
#         BNM_out.append(out)
#
#         # out = self.up6(down6, out)
#         BNM_out.append(out)
#         out = self.up5(down5, out)
#         BNM_out.append(out)
#         out = self.up4(down4, out)
#         BNM_out.append(out)
#         out = self.up3(down3, out)
#         BNM_out.append(out)
#         out = self.up2(down2, out)
#         BNM_out.append(out)
#         out = self.up1(down1, out)
#         BNM_out.append(out)
#         #1024
#
#         out = self.classify(out)
#         out = torch.squeeze(out, dim=1)
#         new_out = []
#         for idx, every_out in enumerate(BNM_out):
#             tmp = F.upsample(every_out, size=(img_size, img_size), mode='bilinear')
#             new_out.append(tmp)
#         new_out = torch.cat(new_out,1)
#         new_out = new_out*self.weights
#         # print new_out.size()
#         new_out = self.boundry_classify(new_out)
#         return [out, new_out]
#
# class BNM_1 (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(BNM_1, self).__init__()
#
#         #1024
#         self.down1 = StackEncoder(  3,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
#         self.down2 = StackEncoder( 24,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.down6 = StackEncoder(512,  768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 ),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(768,  768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  64,  64,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         self.up1 = StackDecoder(  24,  24,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512
#         # self.weights = Variable(torch.randn([1024,1024])).cuda()
#         self.classify = nn.Sequential(
#             nn.Conv2d(24, 1, kernel_size=1, padding=0, stride=1, bias=True),
#             nn.Sigmoid(),
#         )
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#         out = self.up1(down1, out)
#
#         # out = nn.Sigmoid()(self.weights*out)
#         #1024
#
#         out = self.classify(out)
#         # print out.size()
#         # out = torch.squeeze(out, dim=1)
#         return [out]
#
#
# class BNM_2(nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(BNM_2, self).__init__()
#
#         # 1024
#         self.down1 = StackEncoder(3, 24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
#         self.down2 = StackEncoder(24, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
#         self.down3 = StackEncoder(64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
#         self.down4 = StackEncoder(128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.down5 = StackEncoder(256, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.down6 = StackEncoder(512, 768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(768, 768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder(128, 128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
#         self.up2 = StackDecoder(64, 64, 24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
#         self.up1 = StackDecoder(24, 24, 24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
#         self.weights = Variable(torch.randn([1024, 1024])).cuda()
#         self.classify = nn.Sequential(
#             nn.Conv2d(24, 1, kernel_size=1, padding=0, stride=1, bias=True),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         out = x  # ;print('x    ',x.size())
#         #
#         down1, out = self.down1(out)  ##;print('down1',down1.size())  #256
#         down2, out = self.down2(out)  # ;print('down2',down2.size())  #128
#         down3, out = self.down3(out)  # ;print('down3',down3.size())  #64
#         down4, out = self.down4(out)  # ;print('down4',down4.size())  #32
#         down5, out = self.down5(out)  # ;print('down5',down5.size())  #16
#         down6, out = self.down6(out)  # ;print('down6',down6.size())  #8
#         pass  # ;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#         out = self.up1(down1, out)
#
#         out = nn.Sigmoid()(self.weights * out)
#         # 1024
#
#         out = self.classify(out)
#         # out = torch.squeeze(out, dim=1)
#         return [out]
#
# class BNM_3(nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(BNM_3, self).__init__()
#
#         # 1024
#         self.down1 = StackEncoder(3, 24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
#         self.down2 = StackEncoder(24, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
#         self.down3 = StackEncoder(64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
#         self.down4 = StackEncoder(128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.down5 = StackEncoder(256, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         # self.down6 = StackEncoder(512, 768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(512, 512, kernel_size=3, padding=1, stride=1),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         # self.up6 = StackDecoder(768, 768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder(128, 128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
#         self.up2 = StackDecoder(64, 64, 24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
#         self.up1 = StackDecoder(24, 24, 24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
#
#         self.weights = Variable(torch.randn([1024,1024])).cuda()
#         self.classify = nn.Sequential(
#             nn.Conv2d(24, 1, kernel_size=1, padding=0, stride=1, bias=True),
#             nn.Sigmoid(),
#         )
#         self.conv = nn.Conv2d(24, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.conv_1 = nn.Conv2d(24, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.conv_2 = nn.Conv2d(64, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.conv_3 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.conv_4 = nn.Conv2d(256, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.conv_5 = nn.Conv2d(512, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         # self.conv_6 = nn.Conv2d(768, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#
#     def forward(self, x):
#         out = x  # ;print('x    ',x.size())
#         _,_,img_shape,_ = x.size()
#         #
#         down1, out = self.down1(out)  ##;print('down1',down1.size())  #256
#         down2, out = self.down2(out)  # ;print('down2',down2.size())  #128
#         down3, out = self.down3(out)  # ;print('down3',down3.size())  #64
#         down4, out = self.down4(out)  # ;print('down4',down4.size())  #32
#         down5, out = self.down5(out)  # ;print('down5',down5.size())  #16
#         # down6, out = self.down6(out)  # ;print('down6',down6.size())  #8
#         pass  # ;print('out  ',out.size())
#
#         out_5 = self.center(out)
#         # out_5 = self.up6(down6, out_6)
#         out_4 = self.up5(down5, out_5)
#         out_3 = self.up4(down4, out_4)
#         out_2 = self.up3(down3, out_3)
#         out_1 = self.up2(down2, out_2)
#         out = self.up1(down1, out_1)
#
#         out_1 = F.upsample(out_1, size=(img_shape, img_shape), mode='bilinear')
#         out_2 = F.upsample(out_2, size=(img_shape, img_shape), mode='bilinear')
#         out_3 = F.upsample(out_3, size=(img_shape, img_shape), mode='bilinear')
#         out_4 = F.upsample(out_4, size=(img_shape, img_shape), mode='bilinear')
#         out_5 = F.upsample(out_5, size=(img_shape, img_shape), mode='bilinear')
#
#         out = self.conv(out)
#
#         out_1 = self.conv_1(out_1)
#
#         out_2 = self.conv_2(out_2)
#
#         out_3 = self.conv_3(out_3)
#
#         out_4 = self.conv_4(out_4)
#
#         out_5 = self.conv_5(out_5)
#
#         out = torch.cat([out, out_1, out_2, out_3, out_4, out_5],dim=1)
#         out = self.weights*out
#         out = nn.Sigmoid()(out)
#         out = self.classify(out)
#
#         return [out]
#
# class UNet1024 (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(UNet1024, self).__init__()
#
#         #1024
#         self.down1 = StackEncoder(  3,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
#         self.down2 = StackEncoder( 24,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.down6 = StackEncoder(512,  768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 ),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(768,  768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  64,  64,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         self.up1 = StackDecoder(  24,  24,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512
#         self.classify = nn.Conv2d(24, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#         out = self.up1(down1, out)
#         #1024
#
#         out = self.classify(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
#
# class UNet1024_kernel (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(UNet1024_kernel, self).__init__()
#
#         #1024
#         self.down1 = StackEncoder(  3,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
#         self.down2 = StackEncoder( 24,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.down6 = StackEncoder(512,  768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 ),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(768,  768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  64,  64,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         self.up1 = StackDecoder(  24,  24,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512
#         self.classify = nn.Conv2d(24, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.classify1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#         out = self.up1(down1, out)
#         #1024
#
#         out = self.classify(out)
#         out = self.classify1(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
#
# class Multi_Model (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(Multi_Model, self).__init__()
#
#         #1024
#         self.input_1 = StackEncoder(3, 3, kernel_size=7)  # 512
#         self.input_2 = StackEncoder(3, 3, kernel_size=7)  # 512
#         self.input_3 = StackEncoder(3, 3, kernel_size=7)  # 512
#
#         self.down1 = StackEncoder(  9,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
#         self.down2 = StackEncoder( 24,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.down6 = StackEncoder(512,  768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 ),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(768,  768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  64,  64,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         self.up1 = StackDecoder(  24,  24,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512
#         self.classify = nn.Conv2d(24, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#
#     def forward(self, x):
#         out_1,out_2,out_3 = x                       #;print('x    ',x.size())
#         _, out_1 = self.input_1(out_1)
#         _, out_2 = self.input_2(out_2)
#         _, out_3 = self.input_3(out_3)
#         out = torch.cat((out_1, out_2, out_3), dim=1)
#
#                                       #
#         down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#         out = self.up1(down1, out)
#         #1024
#
#         out = self.classify(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
# class Small (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(Small, self).__init__()
#
#         #1024
#         self.down1 = StackEncoder(  3,   32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   #512
#         self.down2 = StackEncoder( 32,   128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   #256
#         self.down3 = StackEncoder( 128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   #128
#         self.down4 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   # 64
#         # self.down5 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   # 32
#         # self.down6 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   # 16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(512, 512, kernel_size=3, padding=1, stride=1 ),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         # self.up6 = StackDecoder(256,  256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 16
#         # self.up5 = StackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 16
#         self.up4 = StackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 16
#         self.up3 = StackDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 32
#         self.up2 = StackDecoder(128, 128, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 32
#         self.up1 = StackDecoder(32, 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 256
#         self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         # down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         # down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         # pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         # out = self.up6(down6, out)
#         # out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#         out = self.up1(down1, out)
#         #1024
#
#         out = self.classify(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
# class MobileNet (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(MobileNet, self).__init__()
#
#         #1024
#         self.down1 = MobileNetEncoder(  3,   32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   #512
#         self.down2 = MobileNetEncoder( 32,   128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   #256
#         self.down3 = MobileNetEncoder( 128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   #128
#         self.down4 = MobileNetEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   # 64
#         # self.down5 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   # 32
#         # self.down6 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)   # 16
#
#         self.center = nn.Sequential(
#             MobileNet_block(512, 512, kernel_size=3, padding=1, stride=1 ),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         # self.up6 = StackDecoder(256,  256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 16
#         # self.up5 = StackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 16
#         self.up4 = MobileNetDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 16
#         self.up3 = MobileNetDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 32
#         self.up2 = MobileNetDecoder(128, 128, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 32
#         self.up1 = MobileNetDecoder(32, 32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm, num_groups=8)  # 256
#         self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         # down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         # down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         # pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         # out = self.up6(down6, out)
#         # out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#         out = self.up1(down1, out)
#         #1024
#
#         out = self.classify(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
#
# class Patch_Model (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(Patch_Model, self).__init__()
#
#         #1024
#         self.down1 = StackEncoder(  3,   12, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
#         self.down2 = StackEncoder( 12,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
#         self.down3 = StackEncoder( 24,  36, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down4 = StackEncoder(36,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down5 = StackEncoder(64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.down6 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(256, 256, kernel_size=3, padding=1, stride=1 ),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(256,  256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 128, 128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 64, 64, 36, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 36, 36,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  24,  24,  12, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         self.up1 = StackDecoder(  12,  12,  12, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512
#         self.classify = nn.Conv2d(12, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         # pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#         out = self.up1(down1, out)
#         #1024
#
#         out = self.classify(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
#
# # 512x512
# class UNet512 (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(UNet512, self).__init__()
#
#         #1024
#         self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #64
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #32
#         self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
#             #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
#         )
#
#         # 16
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#         down2,out = self.down2(out)   #;print('down2',down2.size())
#         down3,out = self.down3(out)   #;print('down3',down3.size())
#         down4,out = self.down4(out)   #;print('down4',down4.size())
#         down5,out = self.down5(out)   #;print('down5',down5.size())
#         down6,out = self.down6(out)   #;print('down6',down6.size())
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#
#         out = self.classify(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
#
# class UNet512_kernel (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(UNet512_kernel, self).__init__()
#
#         #1024
#         self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #64
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #32
#         self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 ),
#             #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
#         )
#
#         # 16
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         self.classify = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.classify1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#         down2,out = self.down2(out)   #;print('down2',down2.size())
#         down3,out = self.down3(out)   #;print('down3',down3.size())
#         down4,out = self.down4(out)   #;print('down4',down4.size())
#         down5,out = self.down5(out)   #;print('down5',down5.size())
#         down6,out = self.down6(out)   #;print('down6',down6.size())
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#
#         out = self.classify(out)
#         out = self.classify1(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
#
# class GroupNorm (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(GroupNorm, self).__init__()
#
#         #1024
#         self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)    #256
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)    #128
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)    #64
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)    #32
#         self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)    #16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn) ,
#             #ConvBnRelu2d(2048, 1024, kernel_size=3, padding=1, stride=1 ),
#         )
#
#         # 16
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
#         self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#         down2,out = self.down2(out)   #;print('down2',down2.size())
#         down3,out = self.down3(out)   #;print('down3',down3.size())
#         down4,out = self.down4(out)   #;print('down4',down4.size())
#         down5,out = self.down5(out)   #;print('down5',down5.size())
#         down6,out = self.down6(out)   #;print('down6',down6.size())
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#
#         out = self.classify(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
#
# # 256x256
# class UNet256 (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(UNet256, self).__init__()
#
#         #256
#         self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#         self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8
#
#         self.center = nn.Sequential(
#             #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
#             ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         # self.classify = nn.Sequential(
#         #     nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True),
#         #     nn.Sigmoid(),
#         # )
#         self.classify = nn.Conv2d(32, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#
#         out = self.classify(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
class UNet256_kernel(nn.Module):
    def __init__(self, n_classes, bn=False, BatchNorm=False):
        super(UNet256_kernel, self).__init__()

        #256
        self.down2 = StackEncoder(1,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
        self.down3 = StackEncoder(64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
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
        self.softmax1 = nn.Softmax(dim=1)

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
        out = self.softmax1(out)
        return [out]


# class UNet256_kernel_dgf (nn.Module):
#     """
#     input should contain x_l and x_h when using guided filter,which is
#     different from other models
#     """
#     def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
#         super(UNet256_kernel_dgf, self).__init__()
#
#         #256
#         self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#         self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8
#
#         self.center = nn.Sequential(
#             #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
#             ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         # self.classify = nn.Sequential(
#         #     nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True),
#         #     nn.Sigmoid(),
#         # )
#         self.classify = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.classify1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#         #guided filter
#         self.gf = FastGuidedFilter(radius, eps)
#         self.guided_map = nn.Sequential(
#             nn.Conv2d(3, cn, 1, bias=False),
#             AdaptiveNorm(cn),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(cn, 1, 1)
#         )
#         self.guided_map.apply(weights_init_identity)
#
#
#     def forward(self, x,x_h):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#
#         out = self.classify(out)
#         out = self.classify1(out)
#         out = torch.squeeze(out, dim=1)
#
#         out = self.gf(self.guided_map(x), out, self.guided_map(x_h))
#
#         return [out]
#
#
# class UNet256_kernel_label (nn.Module):
#     """
#     This model aims to verify the feasibility of Guided Map
#     input should contain x_l and x_h when using guided filter,which is
#     different from other models
#     Author: Shihao Zhang
#     """
#     def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
#         super(UNet256_kernel_label, self).__init__()
#
#         #256
#         self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#         self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8
#
#         self.center = nn.Sequential(
#             #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
#             ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         # self.classify = nn.Sequential(
#         #     nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True),
#         #     nn.Sigmoid(),
#         # )
#         self.classify = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.classify1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#         #guided filter
#         self.gf = FastGuidedFilter(radius, eps)
#         # self.guided_map = nn.Sequential(
#         #     nn.Conv2d(3, cn, 1, bias=False),
#         #     AdaptiveNorm(cn),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     nn.Conv2d(cn, 1, 1)
#         # )
#         # self.guided_map.apply(weights_init_identity)
#
#
#     def forward(self, x,l_h,l_l):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#
#         out = self.classify(out)
#         out = self.classify1(out)
#         out = torch.squeeze(out, dim=1)
#
#
#         l_h=torch.unsqueeze(l_h,1)
#         l_l = torch.unsqueeze(l_l, 1)
#         l_h = l_h.float()
#         l_l = l_l.float()
#
#
#         out = self.gf(l_l, out, l_h)
#
#         return [out]
#
#
# class UNet256_kernel_figure (nn.Module):
#     """
#     This model aims to verify the feasibility of Guided Map, and this model do not contain guided map
#     Input should contain x_l and x_h when using guided filter,which is
#     different from other models
#     Author: Shihao Zhang
#     Data: 2018/10/23
#     """
#     def __init__(self, n_classes, radius=5, eps=1e-4, cn=15, bn=False, BatchNorm=False):
#         super(UNet256_kernel_figure, self).__init__()
#
#         #256
#         self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#         self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8
#
#         self.center = nn.Sequential(
#             #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
#             ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         # self.classify = nn.Sequential(
#         #     nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True),
#         #     nn.Sigmoid(),
#         # )
#         self.classify = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.classify1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#         #guided filter
#         self.gf = FastGuidedFilter(radius, eps)
#         # self.guided_map = nn.Sequential(
#         #     nn.Conv2d(3, cn, 1, bias=False),
#         #     AdaptiveNorm(cn),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     nn.Conv2d(cn, 1, 1)
#         # )
#         # self.guided_map.apply(weights_init_identity)
#
#
#     def forward(self, x,f_l,f_h):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#
#         out = self.classify(out)
#         out = self.classify1(out)
#         out = torch.squeeze(out, dim=1)
#
#         f_h=torch.unsqueeze(f_h,1)
#         f_l = torch.unsqueeze(f_l, 1)
#
#         # f_h = f_h.float()
#         # f_l = f_l.float()
#
#         out = self.gf(f_l, out, f_h)
#
#         return [out]
#
#
# # 128x128
# class UNet128 (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(UNet128, self).__init__()
#
#         #128
#         self.down3 = StackEncoder( 3,   128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#         self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.classify = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())6
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.classify(out)
#         # print out.size()
#         out = torch.squeeze(out, dim=1)
#         # print out.size()
#         return [out]
#
# class UNet512_SideOutput(nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(UNet512_SideOutput, self).__init__()
#
#         #1024
#         self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #64
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #32
#         self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
#         )
#
#         # 16
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         self.classify_1 = nn.Conv2d(32, n_classes , kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify_2 = nn.Conv2d(64, n_classes , kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify_3 = nn.Conv2d(128, n_classes , kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify_4 = nn.Conv2d(256, n_classes , kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify_5 = nn.Conv2d(512, n_classes , kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify_6 = nn.Conv2d(1024, n_classes , kernel_size=1, padding=0, stride=1, bias=True)
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#         down2,out = self.down2(out)   #;print('down2',down2.size())
#         down3,out = self.down3(out)   #;print('down3',down3.size())
#         down4,out = self.down4(out)   #;print('down4',down4.size())
#         down5,out = self.down5(out)   #;print('down5',down5.size())
#         down6,out = self.down6(out)   #;print('down6',down6.size())
#         pass                          #;print('out  ',out.size())
#
#         out_6 = self.center(out)
#         out_5 = self.up6(down6, out_6)
#         out_4 = self.up5(down5, out_5)
#         out_3 = self.up4(down4, out_4)
#         out_2 = self.up3(down3, out_3)
#         out_1 = self.up2(down2, out_2)
#
#         out_1 = self.classify_1(out_1)
#         out_1 = torch.squeeze(out_1, dim=1)
#
#         out_2 = self.classify_2(out_2)
#         out_2 = torch.squeeze(out_2, dim=1)
#
#         out_3 = self.classify_3(out_3)
#         out_3 = torch.squeeze(out_3, dim=1)
#
#         out_4 = self.classify_4(out_4)
#         out_4 = torch.squeeze(out_4, dim=1)
#
#         out_5 = self.classify_5(out_5)
#         out_5 = torch.squeeze(out_5, dim=1)
#
#         out_6 = self.classify_6(out_6)
#         out_6 = torch.squeeze(out_6, dim=1)
#         return [out_1,out_2,out_3,out_4,out_5,out_6]
#
# # 1024x1024
# class UNet1024_SideOutput(nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(UNet1024_SideOutput, self).__init__()
#
#         #1024
#         self.down1 = StackEncoder(  3,   24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #512
#         self.down2 = StackEncoder( 24,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #256
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.down6 = StackEncoder(512,  768, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 , is_bn=bn),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(768,  768, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  64,  64,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         self.up1 = StackDecoder(  24,  24,  24, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #512
#         self.classify = nn.Conv2d(24, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify_1 = nn.Conv2d(24, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify_2 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify_3 = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify_4 = nn.Conv2d(256, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify_5 = nn.Conv2d(512, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#         self.classify_6 = nn.Conv2d(768, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#         self.conv_1 = nn.Conv2d(24, 24, kernel_size=3, padding=1, stride=1, bias=True)
#         self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=True)
#         self.conv_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=True)
#         self.conv_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=True)
#         self.conv_5 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=True)
#         self.conv_6 = nn.Conv2d(768, 768, kernel_size=3, padding=1, stride=1, bias=True)
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         out_6 = self.center(out)
#         out_5 = self.up6(down6, out_6)
#         out_4 = self.up5(down5, out_5)
#         out_3 = self.up4(down4, out_4)
#         out_2 = self.up3(down3, out_3)
#         out_1 = self.up2(down2, out_2)
#         out = self.up1(down1, out_1)
#
#         # out = self.conv_1(out)
#         out = self.classify_1(out)
#         out = torch.squeeze(out, dim=1)
#
#         out_1 = self.conv_1(out_1)
#         out_1 = self.classify_1(out_1)
#         out_1 = torch.squeeze(out_1, dim=1)
#
#         out_2 = self.conv_2(out_2)
#         out_2 = self.classify_2(out_2)
#         out_2 = torch.squeeze(out_2, dim=1)
#
#         out_3 = self.conv_3(out_3)
#         out_3 = self.classify_3(out_3)
#         out_3 = torch.squeeze(out_3, dim=1)
#
#         out_4 = self.conv_4(out_4)
#         out_4 = self.classify_4(out_4)
#         out_4 = torch.squeeze(out_4, dim=1)
#
#         out_5 = self.conv_5(out_5)
#         out_5 = self.classify_5(out_5)
#         out_5 = torch.squeeze(out_5, dim=1)
#
#         # out_6 = self.classify_6(out_6)
#         # out_6 = torch.squeeze(out_6, dim=1)
#         return [out, out_1, out_2, out_3, out_4, out_5]
#
#
#
# # class resnet_50(nn.Module):
# #     def __init__(self, n_classes, pretrain=False, img_size=512):
# #         super(resnet_50, self).__init__()
# #         self.resnet = resnet50(pretrained=pretrain)
# #         self.center = nn.Sequential(
# #             ConvBnRelu2d(2048, 2048, kernel_size=3, padding=1, stride=1),
# #         )
# #         self.up4 = ResStackDecoder(2048, 2048, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
# #         self.up3 = ResStackDecoder(1024, 1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
# #         self.up2 = ResStackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
# #         self.up1 = ResStackDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
# #         self.classify = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
# #         self.img_size = img_size
# #
# #     def forward(self, x):
# #         layer1, layer2, layer3, layer4 = self.resnet(x)
# #         center = self.center(layer4)
# #         out_4 = self.up4(layer4, center)
# #         out_3 = self.up3(layer3, out_4)
# #         out_2 = self.up2(layer2, out_3)
# #         out_1 = self.up1(layer1, out_2)
# #         out = F.upsample(out_1, size=(self.img_size,self.img_size), mode='bilinear')
# #         out = self.classify(out)
# #         return [out]
# #
# # class resnet_dense(nn.Module):
# #     def __init__(self, n_classes, pretrain=False, img_size=512):
# #         super(resnet_dense, self).__init__()
# #         self.resnet = resnet50(pretrained=pretrain)
# #         self.center = nn.Sequential(
# #             ConvBnRelu2d(2048, 2048, kernel_size=3, padding=1, stride=1),
# #         )
# #         self.up4 = ResStackDecoder(2048, 2048, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
# #         self.up3 = ResStackDecoder(1024, 1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
# #         self.up2 = ResStackDecoder(512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
# #         self.up1 = ResStackDecoder(256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
# #         self.classify = nn.Conv2d(128, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
# #         self.classify1 = nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1, bias=True)
# #         self.classify2 = nn.Conv2d(512, 128, kernel_size=1, padding=0, stride=1, bias=True)
# #         self.classify3 = nn.Conv2d(1024, 128, kernel_size=1, padding=0, stride=1, bias=True)
# #         self.classify4 = nn.Conv2d(2048, 128, kernel_size=1, padding=0, stride=1, bias=True)
# #         self.img_size = img_size
# #
# #     def forward(self, x):
# #         layer1, layer2, layer3, layer4 = self.resnet(x)
# #         center = self.center(layer4)
# #         out_4 = self.up4(layer4, center)
# #         out_3 = self.up3(layer3, out_4)
# #         out_2 = self.up2(layer2, out_3)
# #         out_1 = self.up1(layer1, out_2)
# #         layer1 = self.classify1(layer1)
# #         layer2 = self.classify2(layer2)
# #         layer3 = self.classify3(layer3)
# #         layer4 = self.classify4(layer4)
# #
# #         layer1 = F.upsample(layer1, size=(self.img_size, self.img_size), mode='bilinear')
# #         layer2 = F.upsample(layer2, size=(self.img_size, self.img_size), mode='bilinear')
# #         layer3 = F.upsample(layer3, size=(self.img_size, self.img_size), mode='bilinear')
# #         layer4 = F.upsample(layer4, size=(self.img_size, self.img_size), mode='bilinear')
# #         out = F.upsample(out_1, size=(self.img_size,self.img_size), mode='bilinear')
# #         out = (out + layer1 + layer2 + layer3 + layer4)/5
# #         out = self.classify(out)
# #         return [out]
#
#
#
# class UpsampleConvLayer(torch.nn.Module):
#     """UpsampleConvLayer
#     Upsamples the input and then does a convolution. This method gives better results
#     compared to ConvTranspose2d.
#     ref: http://distill.pub/2016/deconv-checkerboard/
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
#         super(UpsampleConvLayer, self).__init__()
#         self.upsample = upsample
#         if upsample:
#             self.upsample_layer = torch.nn.UpsamplingNearest2d(scale_factor=upsample)
#         reflection_padding = kernel_size // 2
#         self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
#         self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
#
#     def forward(self, x):
#         x_in = x
#         if self.upsample:
#             x_in = self.upsample_layer(x_in)
#         out = self.reflection_pad(x_in)
#         out = self.conv2d(out)
#         return out
#
# # 1024x1024
# class UNet1024_deconv(nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(UNet1024_deconv, self).__init__()
#
#         #1024
#         self.down1 = StackEncoder(  3,   24, kernel_size=7, dilation=2)   #512
#         self.down2 = StackEncoder( 24,   64, kernel_size=3, dilation=2)   #256
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, dilation=2)   #128
#         self.down4 = StackEncoder(128,  256, kernel_size=3, dilation=2)   # 64
#         self.down5 = StackEncoder(256,  512, kernel_size=3, dilation=2)   # 32
#         self.down6 = StackEncoder(512,  768, kernel_size=3, dilation=2)   # 16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 , is_bn=bn),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = Decoder(768,  768, 512, kernel_size=3, dilation=2)  # 16
#         self.up5 = Decoder( 512, 512, 256, kernel_size=3, dilation=2)  # 32
#         self.up4 = Decoder( 256, 256, 128, kernel_size=3, dilation=2)  # 64
#         self.up3 = Decoder( 128, 128,  64, kernel_size=3, dilation=2)  #128
#         self.up2 = Decoder(  64,  64,  24, kernel_size=3, dilation=2)  #256
#         self.up1 = Decoder(  24,  24,  24, kernel_size=3, dilation=2)  #512
#         self.classify = nn.Conv2d(24, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#         out = self.up1(down1, out)
#         #1024
#
#         out = self.classify(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
#
# # 128x128
# class UNet128_deconv (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(UNet128_deconv, self).__init__()
#
#         #128
#         self.down3 = StackEncoder( 3,   128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#         self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = UpsampleConvLayer(1024, 512, kernel_size=3, stride=2, upsample=2)
#         self.up5 = UpsampleConvLayer(512, 256, kernel_size=3, stride=2, upsample=2)
#         self.up4 = UpsampleConvLayer(256, 128, kernel_size=3, stride=2, upsample=2)
#         self.up3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=2, upsample=2)
#         self.classify = nn.Conv2d(64, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#     def _upsample_add(self, x, y):
#         _,_,H,W = y.size()
#         return F.upsample(x, size=(H,W), mode='bilinear') + y
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self._upsample_add(self.up6(out), down5)
#         out = self._upsample_add(self.up5(out), down4)
#         out = self._upsample_add(self.up4(out), down3)
#         out = self.up3(out)
#         out = self.classify(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
# # 1024x1024
# class FPN_deconv (nn.Module):
#     def __init__(self, n_classes, bn=False, BatchNorm=False):
#         super(FPN_deconv , self).__init__()
#
#         #1024
#         self.down1 = StackEncoder(  3,   24, kernel_size=7, dilation=2)   #512
#         self.down2 = StackEncoder( 24,   64, kernel_size=3, dilation=2)   #256
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, dilation=2)   #128
#         self.down4 = StackEncoder(128,  256, kernel_size=3, dilation=2)   # 64
#         self.down5 = StackEncoder(256,  512, kernel_size=3, dilation=2)   # 32
#         self.down6 = StackEncoder(512,  768, kernel_size=3, dilation=2)   # 16
#
#         self.center = nn.Sequential(
#             ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 , is_bn=bn),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(768,  768, 512, kernel_size=3, dilation=2)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, dilation=2)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, dilation=2)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, dilation=2)  #128
#         self.up2 = StackDecoder(  64,  64,  24, kernel_size=3, dilation=2)  #256
#         self.up1 = StackDecoder(  24,  24,  24, kernel_size=3, dilation=2)  #512
#         self.classify = nn.Conv2d(24, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#     def _mutil_ave_pooling(self, x, kernel_size=[2,3,4]):
#         output = x
#         _, _, H, W = x.size()
#
#         for k in kernel_size:
#             avg_P = F.avg_pool2d(x, kernel_size=k, stride=k)
#             output += F.upsample(avg_P, size=(H, W), mode='bilinear')
#         return output/len(kernel_size)
#
#
#     def forward(self, x):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         # out = self._mutil_ave_pooling(out, kernel_size=[2,3,6])
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#         out = self.up1(down1, out)
#         #1024
#
#         out = self.classify(out)
#         out = torch.squeeze(out, dim=1)
#         return [out]
#
# class BCELoss2d(nn.Module):
#     def __init__(self):
#         super(BCELoss2d, self).__init__()
#         self.bce_loss = nn.BCEWithLogitsLoss()
#         #self.bce_loss = StableBCELoss()
#     def forward(self, logits, labels):
#         logits_flat = logits.view (-1)
#         labels_flat = labels.view(-1)
#         return self.bce_loss(logits_flat, labels_flat)
#
# # if __name__ == '__main__':
# #
# #     CARVANA_HEIGHT = 1280
# #     CARVANA_WIDTH  = 1918
# #     batch_size  = 1
# #     C,H,W = 3,512,512    #3,CARVANA_HEIGHT,CARVANA_WIDTH
# #
# #     num_classes = 4
# #
# #     inputs = torch.randn(batch_size,C,H,W)
# #     labels = torch.FloatTensor(batch_size,H,W).random_(4).type(torch.LongTensor)
# #     lossfunc = nn.NLLLoss2d()
# #
# #     for model in [UNet1024_deconv,UNet128_deconv,UNet128, UNet256, UNet512, UNet1024, UNet512_SideOutput, UNet1024_SideOutput, resnet_50, resnet_dense]:
# #         net = model(n_classes=num_classes).train()
# #         x = Variable(inputs)
# #         y = Variable(labels)
# #         # net = model(n_classes=num_classes).cuda().train()
# #         # x = Variable(inputs.cuda())
# #         # y = Variable(labels.cuda())
# #         logits = net.forward(x)
# #         output_size = len(logits)
# #         if output_size==1:
# #
# #             loss = lossfunc(logits[0], y)
# #             print logits[0].size()
# #             print loss.data[0]
# #             loss.backward()
# #         else:
# #             print 'Side_Output'
#
#
# # ============================================
# # guided filter
# # ----------------------------------------------
#
#
# def weights_init_identity(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         n_out, n_in, h, w = m.weight.data.size()
#         # Last Layer
#         if n_out < n_in:
#             init.xavier_uniform(m.weight.data)
#             return
#
#         # Except Last Layer
#         m.weight.data.zero_()
#         ch, cw = h // 2, w // 2
#         for i in range(n_in):
#             m.weight.data[i, i, ch, cw] = 1.0
#
#     elif classname.find('BatchNorm2d') != -1:
#         init.constant(m.weight.data, 1.0)
#         init.constant(m.bias.data,   0.0)
#
#
# class AdaptiveNorm(nn.Module):
#     def __init__(self, n):
#         super(AdaptiveNorm, self).__init__()
#
#         self.w_0 = nn.Parameter(torch.Tensor([1.0]))
#         self.w_1 = nn.Parameter(torch.Tensor([0.0]))
#
#         self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)
#
#     def forward(self, x):
#         return self.w_0 * x + self.w_1 * self.bn(x)
#
#
# class DeepGuidedFilter(nn.Module):
#     def __init__(self, radius=1, eps=1e-4):
#         super(DeepGuidedFilter, self).__init__()
#
#         self.gf = FastGuidedFilter(radius, eps)
#
#         self.guided_map = nn.Sequential(
#             nn.Conv2d(3, 15, 1, bias=False),
#             AdaptiveNorm(15),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(15, 1, 1)
#         )
#         self.guided_map.apply(weights_init_identity)
#
#     def forward(self, x_lr, y_lr, x_hr):
#         return self.gf(self.guided_map(x_lr), y_lr, self.guided_map(x_hr))
#        # return self.gf(self.guided_map(x_lr), y_lr, self.guided_map(x_hr)).clamp(0, 1)
#
# class UGF(nn.Module):
#     """
#     input should contain x_l and x_h when using guided filter,which is
#     different from other models
#     """
#     def __init__(self, n_classes, radius=5, eps=1e-1, cn=15, bn=False, BatchNorm=False):
#         super(UGF, self).__init__()
#
#         #256
#         self.down2 = StackEncoder(  3,   64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #128
#         self.down3 = StackEncoder( 64,  128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 64
#         self.down4 = StackEncoder(128,  256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 32
#         self.down5 = StackEncoder(256,  512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   # 16
#         self.down6 = StackEncoder(512, 1024, kernel_size=3, bn=bn, BatchNorm=BatchNorm)   #  8
#
#         self.center = nn.Sequential(
#             #ConvBnRelu2d( 512, 1024, kernel_size=3, padding=1, stride=1 ),
#             ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1 , is_bn=bn),
#         )
#
#         # 8
#         # x_big_channels, x_channels, y_channels
#         self.up6 = StackDecoder(1024,1024, 512, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 16
#         self.up5 = StackDecoder( 512, 512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 32
#         self.up4 = StackDecoder( 256, 256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64
#         self.up3 = StackDecoder( 128, 128,  64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #128
#         self.up2 = StackDecoder(  64,  64,  32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  #256
#         # self.classify = nn.Sequential(
#         #     nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True),
#         #     nn.Sigmoid(),
#         # )
#         self.classify = nn.Conv2d(32, n_classes, kernel_size=3, padding=1, stride=1, bias=True)
#         self.classify1 = nn.Conv2d(n_classes, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
#
#         #guided filter
#         self.gf = FastGuidedFilter(radius, eps)
#         self.guided_map = nn.Sequential(
#             nn.Conv2d(3, cn, 1, bias=False),
#             AdaptiveNorm(cn),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(cn, 1, 1)
#         )
#         self.guided_map.apply(weights_init_identity)
#
#
#     def forward(self, x,x_h):
#
#         out = x                       #;print('x    ',x.size())
#                                       #
#         down2,out = self.down2(out)   #;print('down2',down2.size())  #128
#         down3,out = self.down3(out)   #;print('down3',down3.size())  #64
#         down4,out = self.down4(out)   #;print('down4',down4.size())  #32
#         down5,out = self.down5(out)   #;print('down5',down5.size())  #16
#         down6,out = self.down6(out)   #;print('down6',down6.size())  #8
#         pass                          #;print('out  ',out.size())
#
#         out = self.center(out)
#         out = self.up6(down6, out)
#         out = self.up5(down5, out)
#         out = self.up4(down4, out)
#         out = self.up3(down3, out)
#         out = self.up2(down2, out)
#
#         out = self.classify(out)
#         out = self.classify1(out)
#         out = torch.squeeze(out, dim=1)
#
#         out = self.gf(self.guided_map(x), out, self.guided_map(x_h))
#
#         return [out]