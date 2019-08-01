# coding=utf-8

import torch
import numpy as np
from models import *
from models2 import *
# from train import train
from tain_unet import train
import torch.nn as nn
import torch.optim as optim
from dataloader import loadImageList, loaddata
import os
from test import val_model
import torchvision.models as models
import random
from metrics import Metrics
os.environ["CUDA_VISIBLE_DEVICES"] = '6'


fl = open('/home/intern1/qiuzhen/Works/result_for_structure_segment/yin_U_Net_LRS_256_cv2_newdata.txt', 'x')

path = '/data/zhangshihao/ASOCT-new/data/dataset_16_LRS_final'
batchsize = 8
# net = M_Net(4, BatchNorm=True)
net = My_UNet1024([3, 256, 256])
# net = torch.load('/home/intern1/qiuzhen/Works/result_for_structure_segment/yin_U_Net_LRS_256_cv2_newdata.pkl')
# net = UNet256_kernel(4, BatchNorm=True)
# net = torch.load('/home/intern1/qiuzhen/Works/result_for_structure_segment/M_Net_LBO_256_cv2.pkl')
print('model : yin_unet_for_newdata')
# net = models.resnet34(pretrained=True)
# fc_features = net.fc.in_features
# net.fc = nn.Linear(fc_features, 2)
net.cuda()
loss = nn.CrossEntropyLoss(size_average=True).cuda()
optimizer = optim.Adam(net.parameters(), lr=0.01)  #优化方法
image_list, iterper_epo = loadImageList(path, batchsize=batchsize, flag='train')
total = len(image_list)
print('train_data_len:' + str(total))
metric = Metrics(4)
epochs = 500
max = 1.28982
for i in range(epochs):
    net.train()
    metric.reset()
    acc_list = []
    random.shuffle(image_list)
    running_loss = []
    for j in range(iterper_epo):
        if j == (iterper_epo - 1):
            iterlist = image_list[j * batchsize:]
        else:
            iterlist = image_list[j * batchsize: (j + 1) * batchsize]
        img_data, img_label = loaddata(path, iterlist)
        r_loss, correct = train(net, loss, optimizer, img_data, img_label, metric)
        running_loss += [r_loss]
        acc_list += [(int(correct) / (img_label.shape[0] * 256 * 256))]

    mean_iou = metric.compute_final_metrics(iterper_epo)
    avg_loss = np.mean(running_loss)
    acc = np.mean(acc_list)
    fl.write('Epoch : ' + str(i+1) + ' running_loss : ' + str(avg_loss) + '\n')
    fl.write('accuracy : ' + str(acc)[:6] + ' mean_iou : ' + str(mean_iou) + '\n' + '\n')
    print("Epoch %d running_loss=%.3f" % (i+1, avg_loss))
    print('accuracy : ' + str(acc)[:6] + ' mean_iou : ' + str(mean_iou))
    # val_model(path, net, flag=True)
    # torch.save(net, '/home/intern1/qiuzhen/Works/result_for_structure_segment/M_Net_LRS.pkl')
    if i % 3 == 2:
        pixel_loss = val_model(path, net, fl, flag=True)
        if pixel_loss < max:
            max = pixel_loss
            torch.save(net, '/home/intern1/qiuzhen/Works/result_for_structure_segment/yin_U_Net_LRS_256_cv2_newdata.pkl')


# torch.save(net,'/home/intern1/qiuzhen/Works/resnet50.pkl')
fl.close()
print("Finished  Training")
