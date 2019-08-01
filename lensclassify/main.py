# coding=utf-8

import torch
import numpy as np
import model
from model import *
from model import train
import torch.nn as nn
import torch.optim as optim
from dataloader import loadImageList, loaddata
from loadmodel import load_model
import os
from test import val_model
import torchvision.models as models
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '5'

path = '/data/zhangshihao/ASOCT-new/data/dataset_16_LRS_final'
batchsize = 32
# net=load_model('/home/intern1/qiuzhen/Works/light_model_classi3.pkl')
# net = resnext50(4, 32)
# net = ShuffleNet()
# net = Net()
# net = models.vgg11_bn(pretrained=False, num_classes=2)
print('model : resnet34')
net = models.resnet34(pretrained=False, num_classes=5)
# net = models.resnet34(pretrained=True)
# fc_features = net.fc.in_features
# net.fc = nn.Linear(fc_features, 5)
net.cuda()
loss = nn.CrossEntropyLoss(size_average=True).cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  #优化方法
image_list, iterper_epo = loadImageList(path, batchsize=batchsize, flag='train')
total = len(image_list)
print('train_data_len:' + str(total))
last_distance = 2
epochs = 500
for i in range(epochs):
    net.train()
    random.shuffle(image_list)
    accnum = 0
    distance_loss = []
    running_loss = []
    for j in range(iterper_epo):
        if j == (iterper_epo - 1):
            iterlist = image_list[j * batchsize:]
        else:
            iterlist = image_list[j * batchsize: (j + 1) * batchsize]
        img_data, img_label = loaddata(path, iterlist)
        r_loss, num = train(net, loss, optimizer, img_data, img_label)
        running_loss += [r_loss]
        accnum += num
        # for k in range(len(iterlist)):
        #     distance_loss += [abs(predicted[k]-img_label[k])]

        # print(str(np.mean(distance_loss)))
    avg_loss = np.mean(running_loss)
    # avg_distance = np.mean(distance_loss)
    # std_distance = np.std(distance_loss, ddof=1)
    print("Epoch %d running_loss=%.3f" % (i+1, avg_loss))
    print("Accuracy of the prediction on the train dataset : %d %%" % (100 * accnum / total))
    # print("avg_distance:" + str(avg_distance) + " std_distance:" + str(std_distance))
    if i % 3 == 2:
        avg_distance = val_model(path, net)
        if avg_distance < last_distance:
            torch.save(net, '/home/intern1/qiuzhen/Works/for_level/LRS_random_dataset_resnet34_8bit_45_3_nopretrain.pkl')
            last_distance = avg_distance
        # torch.save(net, '/home/intern1/qiuzhen/Works/vgg11_8.pkl')


# torch.save(net,'/home/intern1/qiuzhen/Works/resnet50.pkl')
print("Finished  Training")
