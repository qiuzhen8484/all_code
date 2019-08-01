# coding=utf-8

import torch
import numpy as np
import model
from model import *
from model import train
import torch.nn as nn
import torch.optim as optim
import os
from test import val_model
import torchvision.models as models
import random
import torchvision.transforms as transforms
from asoct import AsoctDataset
from net.resnet import resnet18, resnet10

def adjust_learning_rate(optimizer, epoch):
    update_list = [15, 30, 45, 60]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# set parameters
path = '/data/qiuzhen/cataract_classifi'
batch_size = 64
epochs = 300
best_acc = 0

print('model : no_normalize_shufflenet_avgpool64')
# net = resnet18(pretrained=False, num_classes=2)
# net = resnet10(pretrained=False, num_classes=2)
# net = torch.load('./model/shufflenet_bat_size48_acc43.717277486910994.pkl')
# net = ShuffleNet(in_channels=1, num_classes=2)
net = ShuffleNet_avgpool(in_channels=1, num_classes=2)
net.cuda()
loss = nn.CrossEntropyLoss(size_average=True).cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# training dataset
# transform_train = transforms.Compose([
#     transforms.Resize((600,200)),  # 保持长宽比不变，最短边为400，（h,w）
#     transforms.RandomRotation(10),
#     transforms.RandomCrop((540,180), padding=0),  # 先四周填充0，在吧图像随机裁剪成h*w
#     transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
# ])
transform_train = transforms.Compose([
    transforms.Resize((600, 200)),  # 保持长宽比不变，最短边为400，（h,w）
    transforms.RandomRotation(10),
    transforms.RandomCrop((576, 192), padding=0),  # 先四周填充0，在吧图像随机裁剪成h*w
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])
train_dataset = AsoctDataset(root=path, train=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size,
    shuffle=True)

# transform_test = transforms.Compose([
#     transforms.Resize((540,180)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
transform_test = transforms.Compose([
    transforms.Resize((576, 192)),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = AsoctDataset(root=path, train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

for i in range(epochs):
    net.train()
    accnum = 0.0
    running_loss = []
    total = 0.0
    adjust_learning_rate(optimizer, i)

    for j, (img_data, img_label) in enumerate(train_loader):
        r_loss, num = train(net, loss, optimizer, img_data, img_label)
        running_loss += [r_loss]
        total += img_label.size(0)
        accnum += num
    avg_loss = np.mean(running_loss)
    print("Epoch %d running_loss=%.3f" % (i+1, avg_loss))
    print("Accuracy of the prediction on the train dataset : %f %%" % (100 * np.float(accnum) / np.float(total)))

    # 验证模型
    acc = val_model(net, test_loader)
    if acc > best_acc:
        print('saving the best model!')
        torch.save(net, './model_forcleandata/no_normalize_shufflenet_avgpool64_acc' + str(acc) + '.pkl')
        best_acc = acc

    print('Val acc is : %.04f, best acc is : %.04f' % (acc, best_acc))
    print('================================================')


# torch.save(net,'/home/intern1/qiuzhen/Works/resnet50.pkl')
print("Finished  Training")
