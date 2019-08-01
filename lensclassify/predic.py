import torch
import numpy as np
from dataloader import loadImageList, loadpreddata
import model
from torch.autograd import Variable
from loadmodel import load_model
import os
from shutil import copyfile
from dataloader import testdata,Loaddata
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

path = '/home/intern1/qiuzhen/Works/test/datasets/jpg'
path1 = '/home/intern1/qiuzhen/Works/test/outputs'

batchsize = 32
net = load_model('/home/intern1/qiuzhen/Works/resnetforclassi.pkl')
net.eval()

image_list, iterper_epo = loadImageList(path, batchsize=batchsize, flag='pred')
for i in range(iterper_epo):
    if i == (iterper_epo-1):
        iterlist = image_list[i*batchsize:]
    else:
        iterlist = image_list[i*batchsize : (i+1)*batchsize]
    img_data = loadpreddata(path, iterlist)
    outputs = net.forward(Variable(img_data.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    # print(predicted.shape)
    # print(predicted[0], predicted[1])
    for j in range(len(predicted)):
        if predicted[j] == 1:
            copyfile(os.path.join(path, iterlist[j]), os.path.join(path1, iterlist[j]))


