import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

import cv2
import numpy as np
import random
import os
import cPickle as pkl
import argparse
import time
from skimage import segmentation


from utils import MSE_pixel_loss
from utils import get_cur_model
from utils import get_new_data
from utils import crop_boundry
from utils import calculate_Accuracy
from utils import get_truth
from segmentation import DecideBoundaryLeftRight
from segmentation import TransformToBinaryImage
from ModelUNetTogether import UNet
import torch.nn.functional as F


models_list = ['UNet128', 'UNet256', 'UNet512', 'UNet1024', 'UNet512_SideOutput', 'UNet1024_SideOutput',
              'resnet_50', 'resnet_dense', 'PSP', 'dense_net', 'UNet128_deconv', 'UNet1024_deconv',
               'FPN18','    FPN_deconv',  'My_UNet', 'Unet',   'Multi_scale',     'ASPP']

torch.manual_seed(1111)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--data_path', type=str, default='./data/dataset',
                    help='dir of the all img')
parser.add_argument('--results', type=str,  default='./visual_results',
                    help='path to save the visualization image')
parser.add_argument('--best_model', type=str,  default='250_1e-08.pth',
                    help='the pretrain model')
parser.add_argument('--flag', type=str,  default='train',
                    help='the pretrain model')

parser.add_argument('--model_id', type=int, default=16,
                    help='the id of choice_model in models_list')
parser.add_argument('--epochs', type=int, default=300,
                    help='the epochs of this run')
parser.add_argument('--batch_size', type=int, default=10,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=1024,
                    help='the train img size')
parser.add_argument('--n_class', type=int, default=4,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--pre_lr', type=float, default=0.000025,
                    help='pretrain model parameters learning rate, eg pretrain_resnet50')
parser.add_argument('--init_clip_max_norm', type=float, default=0.05,
                    help='pretrain model parameters learning rate, eg pretrain_resnet50')

parser.add_argument('--use_gpu', type=bool, default=True,
                    help='dir of the all ori img')
parser.add_argument('--test_every_step', type=bool, default=False,
                    help='test after every train step')
parser.add_argument('--pre_all', type=bool, default=False,
                    help='pretrain the whole model')
parser.add_argument('--pre_part', type=bool, default=True,
                    help='pretrain the pytorch pretrain_model, eg resnet50 and resnet18, used in resnet_50 and resnet_dense')
parser.add_argument('--hard_example_train', type=bool, default=False,
                    help='only train the hard example')

args = parser.parse_args()
ori_data = os.path.join(args.data_path,'img')
train_data = os.path.join(args.data_path,str(args.n_class),'train_data')
test_data = os.path.join(args.data_path,str(args.n_class),'test_data')
label_data = os.path.join(args.data_path,str(args.n_class),'train_label')

model_name = models_list[args.model_id]
# model = get_cur_model(model_name, n_classes=args.n_class, pretrain=args.pre_part)
model = UNet(n_classes=args.n_class)
def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []
    b.append(model.down2)
    b.append(model.down3)
    b.append(model.down4)
    b.append(model.down5)
    b.append(model.down6)

    b.append(model.up6)
    b.append(model.up5)
    b.append(model.up4)
    b.append(model.up3)
    b.append(model.up2)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.classify.parameters())
    for j in range(len(b)):
        for i in b[j]:
            yield i
#
# model_name = models_list[args.model_id]
# model = get_cur_model(model_name, n_classes=args.n_class, pretrain=args.pre_part)
# save_model_path = os.path.join('./models',model_name)

# lossfunc = nn.CrossEntropyLoss()
lossfunc = nn.NLLLoss2d()
softmax_2d = nn.Softmax2d()
# log_softmax = nn.LogSoftmax()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
if args.use_gpu:
    model.cuda()
if args.pre_all:
    model_path = os.path.join('models',model_name,args.best_model)
    model.load_state_dict(torch.load(model_path))

img_list = os.listdir(train_data)

# the train.pkl save the boundry information of all images
with open(os.path.join(args.data_path, 'train.pkl')) as f:
    img_infor = pkl.load(f)

if args.hard_example_train:
    with open('./logs/hard_example.pkl') as f:
        img_list = pkl.load(f)
EPS = 1e-12
mean_loss = 100
weight_decay = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# optimizer = torch.optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.lr},
#                               {'params': get_10x_lr_params(model), 'lr':  args.lr}], lr=args.lr, momentum=0.9,
#                               weight_decay=weight_decay)

for epoch in range(args.epochs):
    print 'This model is %s_%s_%s'%(model_name,args.n_class,args.img_size)
    model.train()
    loss_list = []
    random.shuffle(img_list)

    if (epoch % 50 == 0) and epoch != 0:
        args.lr /= 10
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.lr},
        #                        {'params': get_10x_lr_params(model), 'lr': args.lr}], lr=args.lr, momentum=0.9,
        #                       weight_decay=weight_decay)

    for i, path in enumerate(img_list):
        img, label_list = get_new_data(args.data_path, path, img_size=args.img_size, n_classes = args.n_class)
        gt, tmp_gt = label_list[0]
        model.zero_grad()
        if model_name=='Multi_scale':
            out = model.MutiScaleForward(img)
            out_ori = softmax_2d(out[0])
            # out_75 = torch.log(softmax_2d(out[1])+EPS)
            out_50 = softmax_2d(out[1])
            tmp_out = F.upsample(out_50, size=(args.img_size,args.img_size), mode='bilinear')
            out = 0.8*out_ori + 0.2*tmp_out

            out_50 = torch.log(out_50 + EPS)
            out_ori = torch.log(out_ori + EPS)
            out = torch.log(out + EPS)
            _, label_list = get_new_data(args.data_path, path, img_size=int(args.img_size*0.5), n_classes=args.n_class)
            gt_50, _ = label_list[0]

            # _, label_list = get_new_data(args.data_path, path, img_size=int(args.img_size * 0.75), n_classes=args.n_class)
            # gt_75, _ = label_list[0]

            loss_ori = lossfunc(out_ori, gt)
            loss_50 = lossfunc(out_50, gt_50)
            # loss_75 = lossfunc(out_75, gt_75)
            loss = lossfunc(out, gt)
            # loss += loss_ori + loss_50 + loss_75
            loss += loss_ori + loss_50

        elif model_name=='ASPP':
            out = model.ASPPforward(img)
            out = torch.log(softmax_2d(out) + EPS)
            loss = lossfunc(out, gt)
        else:
            out = model(img)
            out = torch.log(softmax_2d(out)+EPS)
            loss = lossfunc(out, gt)

        ppi = np.argmax(out.cpu().data.numpy(), 1).reshape((args.img_size, args.img_size))

        new_Image = ppi.astype(np.uint8)

        tmp_out = new_Image.reshape([-1])
        tmp_gt = tmp_gt.reshape([-1])

        confusion = np.zeros([args.n_class, args.n_class])
        for idx in xrange(len(tmp_gt)):
            confusion[tmp_gt[idx], tmp_out[idx]] += 1
        meanIU, pixelAccuracy, meanAccuracy, classAccuracy = calculate_Accuracy(confusion)
        print('epoch_batch: {:d}_{:d} | loss: {:.2f}  | meanIOU: {:.2f}'.format(epoch, i, loss.data[0], meanIU))

        loss.backward()
        optimizer.step()

        with open('./logs/%s_log.txt' % (model_name), 'a+') as f:
            log_str = "%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (epoch, i, loss, meanIU, pixelAccuracy, meanAccuracy, classAccuracy)
            f.writelines(str(log_str) + '\n')

        if epoch % 4 == 0 and i %4== 0 and epoch!=0:
            left, right, top = img_infor[path]['position']
            w, h = img_infor[path]['size'][:2]

            new_Image = ppi.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            new_Image = cv2.morphologyEx(new_Image, cv2.MORPH_CLOSE, kernel)

            ori_img_path = os.path.join(ori_data, path)
            ori_img = cv2.imread(ori_img_path)
            ROI_img = ori_img[top:, left:right, :].copy()
            pred_img = cv2.resize(new_Image.astype(np.uint8), (ROI_img.shape[1], ROI_img.shape[0]),
                                  interpolation=cv2.INTER_LANCZOS4)
            ROI_img = crop_boundry(ROI_img, pred_img)
            alpha_img = ori_img.copy()
            alpha_img[top:, left:right] = ROI_img
            ROI_img = cv2.addWeighted(alpha_img, 0.4, ori_img, 0.6, 0)
            try:
                ROI_img = get_truth(ROI_img, path, [left, right], top)
            except:
                print path

            if not os.path.exists('./visual_results/%s'%model_name):
                os.mkdir('./visual_results/%s'%model_name)
            save_name = './visual_results/%s/label_%s_%s.png' % (model_name,epoch, i)
            cv2.imwrite(save_name, ROI_img)

    MSE_loss = None
    if not os.path.exists('./models/%s'%model_name):
        os.mkdir('./models/%s'%model_name)

    if epoch and epoch%10==0:
        torch.save(model.state_dict(), './models/%s/%s_%s.pth' % (model_name, epoch, args.lr))
        print 'success save every step model'
    if epoch==args.epochs-1:
        torch.save(model.state_dict(), './models/%s/final.pth' % (model_name))
        print 'success save the final model'

