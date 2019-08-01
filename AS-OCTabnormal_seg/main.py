import torch
import numpy as np
from models import *
# from train import train
from tain_unet import train, iou, Dice, DiceLoss
import torch.nn as nn
import torch.optim as optim
from dataloader import loadImageList, loaddata
import os
from test import val_model
import torchvision.models as models
from models2 import *
import random
from metrics import Metrics
from asoct import AsoctDataset
import torchvision.transforms as transforms

def adjust_learning_rate(optimizer, epoch):
    update_list = [30, 60, 100, 120]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


os.environ["CUDA_VISIBLE_DEVICES"] = '6'


# fl = open('./log/yin_unet_256_wlesion_2.txt', 'x')
fl = open('./log/unet_with_multiscale_256_wlesion.txt', 'x')

path = '/data/zhangshihao/AbnormalRegionData/data/16bit_to_8bit_label/refine_xxl/separate/w_lesion'
batchsize = 12
best_dice = 0
# net = M_Net(4, BatchNorm=True)
# net = UNet256_kernel(2, BatchNorm=True)
# net = My_UNet1024([1, 256, 256], 2)
net = U_Net_with_Multi_Scale(1, 2)
# net = torch.load('/home/intern1/qiuzhen/Works/result_for_structure_segment/M_Net_LBO_256_cv2.pkl')
print('model : U_Net_with_multiscale_wlesion')
net.cuda()
loss = nn.CrossEntropyLoss(size_average=True).cuda()
# loss = DiceLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=0.01)  #优化方法

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),  # 保持长宽比不变，最短边为400，（h,w）转
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])
transform_test = transforms.Compose([
    transforms.Resize((256, 256)),  # 保持长宽比不变，最短边为400，（h,w）
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])
train_set = AsoctDataset(root=path, train=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True)

test_set = AsoctDataset(root=path, train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)

epochs = 500

for i in range(epochs):
    adjust_learning_rate(optimizer, i)
    net.train()
    acc_list = []
    iou_list = []
    dice_list = []
    running_loss = []
    for _, (img_data, img_label, img_name) in enumerate(train_loader):
        r_loss, correct, pre_b = train(net, loss, optimizer, img_data, img_label)
        iou_list.append(iou(pre_b, Variable(img_label))[0])
        dice_list.append(Dice(pre_b, Variable(img_label))[0])
        running_loss += [r_loss]
        acc_list += [(int(correct) / (img_label.shape[0] * 256 * 256))]

    mean_iou_v2 = np.mean(iou_list)
    dice = np.mean(dice_list)
    avg_loss = np.mean(running_loss)
    acc = np.mean(acc_list)
    fl.write('Epoch : ' + str(i+1) + ' running_loss : ' + str(avg_loss) + '\n')
    fl.write('mean_iou_v2 : ' + str(mean_iou_v2) + ' dice: ' + str(dice) + '\n')
    fl.write('accuracy : ' + str(acc)[:6] + '\n' + '\n')
    print("Epoch %d running_loss=%.3f" % (i+1, avg_loss))
    print('accuracy : ' + str(acc)[:6] + ' mean_iou_v2: ' + str(mean_iou_v2) + ' dice: ' + str(dice))
    if i % 3 == 2:
        test_dice = val_model(path, net, fl, flag=True, test_loader=test_loader)
        if test_dice > best_dice:
            best_dice = test_dice
            print('best_dice: ' + str(best_dice))
            torch.save(net, './log/U_Net_with_multiscale_256_wlesion.pkl')
        fl.write('best_dice: ' + str(best_dice) + '\n')


fl.close()
print("Finished  Training")
