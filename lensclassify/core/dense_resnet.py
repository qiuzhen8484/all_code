
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np

import os
import random
import cPickle as pkl

from utils import get_new_data
from utils import calculate_Accuracy
from utils import crop_boundry

# from resNet_model import resnet50
# from resNet_model import resnet_up
from resNet_model import dense_resnet


model = dense_resnet(pretrain=True)
# detection = torchvision.datasets.coco.CocoDetection(root='../dataset/cocoapi/coco/val2014/',
#                                        annFile='../dataset/cocoapi/annotations/instances_val2014.json',
#                                        transform=transforms.ToTensor())
# x = Variable(torch.randn([2,3,512,512]))
# y = model(x)
# print y.size()

img_size = 512
n_class = 4
gpu_num = 1
try:
    model.cuda(gpu_num)
except:
    print 'failed to get gpu'
# model.load_state_dict(torch.load('./models/4.pth'))

# lossfunc = nn.BCELoss()
# lossfunc = nn.CrossEntropyLoss()
# lossfunc = nn.NLLLoss2d(weight=weights)
lossfunc = nn.NLLLoss2d()

softmax_2d = nn.Softmax2d()
lr = 0.0001
optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

data_path = './data/dataset'
img_list = os.listdir(os.path.join(data_path, 'train_data'))
min_loss = 100
flag = 0

with open(os.path.join(data_path, 'train.pkl')) as f:
    img_infor = pkl.load(f)

for epoch in range(20):
    torch.save(model.state_dict(), './models/%s_%s.pth' % (epoch, lr))
    loss_list = []
    random.shuffle(img_list)

    if epoch % 10 == 0 and epoch != 0:
        lr /= 10
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    for i, path in enumerate(img_list):

        ## my_Unet
        img, label_list = get_new_data(data_path, path, img_size=img_size, gpu_num=gpu_num)
        gt, tmp_gt = label_list[0]
        model.zero_grad()
        out = model(img)
        out = torch.log(softmax_2d(out))

        loss = lossfunc(out, gt)

        ppi = np.argmax(out.cpu().data.numpy(), 1).reshape((img_size, img_size))

        new_Image = ppi.astype(np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # new_Image = cv2.morphologyEx(new_Image, cv2.MORPH_CLOSE, kernel)
        tmp_out = new_Image.reshape([-1])
        tmp_gt = tmp_gt.reshape([-1])

        confusion = np.zeros([n_class, n_class])
        for idx in xrange(len(tmp_gt)):
            confusion[tmp_gt[idx], tmp_out[idx]] += 1
        meanIU, pixelAccuracy, meanAccuracy, classAccuracy = calculate_Accuracy(confusion)

        loss.backward()


        with open('./logs/log.txt', 'a+') as f:
            log_str = "%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (
            epoch, i, loss, meanIU, pixelAccuracy, meanAccuracy, classAccuracy)
            f.writelines(str(log_str) + '\n')

        if epoch % 2 == 0 and epoch != 0:
            torch.save(model.state_dict(), './models/%s_%s.pth' % (epoch, lr))
            print('success')

        if epoch % 4 == 0 and i %4== 0 and epoch!=0:
            left, right, top = img_infor[path]['position']
            w, h = img_infor[path]['size'][:2]

            new_Image = ppi.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            new_Image = cv2.morphologyEx(new_Image, cv2.MORPH_CLOSE, kernel)

            ori_img_path = os.path.join(data_path, 'img', path)

            ori_img = cv2.imread(ori_img_path)

            ROI_img = ori_img[top:, left:right, :].copy()
            pred_img = cv2.resize(new_Image.astype(np.uint8), (ROI_img.shape[1], ROI_img.shape[0]),
                                  interpolation=cv2.INTER_LANCZOS4)
            ROI_img = crop_boundry(ROI_img, pred_img)
            alpha_img = ori_img.copy()
            alpha_img[top:, left:right] = ROI_img

            ROI_img = cv2.addWeighted(alpha_img, 0.4, ori_img, 0.6, 0)

            save_name = './data/dataset/train_results/label_%s_%s.png' % (epoch, i)
            cv2.imwrite(save_name, ROI_img)
            print('success')

        # print('epoch-i: {:5d,}  |  i: {:3d}  |  loss: {:5.2f}'.format(epoch, i, loss.data[0]))
        print('epoch_batch: {:d}_{:d} | loss: {:.2f}  | meanIOU: {:.2f}'.format(epoch, i, loss.data[0], meanIU))
        optimizer.step()