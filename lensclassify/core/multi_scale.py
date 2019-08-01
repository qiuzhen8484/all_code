import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F

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
from utils import get_multi_scale_data
from utils import crop_boundry
from utils import calculate_Accuracy
from utils import get_truth
from segmentation import DecideBoundaryLeftRight
from segmentation import TransformToBinaryImage


models_list = ['UNet128',   'UNet256',      'UNet512',  'UNet1024',    'UNet512_SideOutput',   'UNet1024_SideOutput',
              'resnet_50',  'resnet_dense', 'PSP',      'dense_net',   'UNet128_deconv',       'UNet1024_deconv',
               'FPN18',     'FPN_deconv',   'multi_scale']

torch.manual_seed(1111)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--data_path', type=str, default='./data/dataset',
                    help='dir of the all img')
parser.add_argument('--results', type=str,  default='./visual_results',
                    help='path to save the visualization image')
parser.add_argument('--best_model', type=str,  default='final.pth',
                    help='the pretrain model')
parser.add_argument('--flag', type=str,  default='train',
                    help='the pretrain model')

parser.add_argument('--model_id', type=int, default=14,
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
model = get_cur_model(model_name, n_classes=args.n_class, pretrain=args.pre_part)
# save_model_path = os.path.join('./models',model_name)
if 'resnet' in model_name:
    params_mask = list(map(id, model.resnet.parameters()))
    base_params = filter(lambda p: id(p) not in params_mask,
                         model.parameters())
    pre_params = filter(lambda p: id(p) in params_mask,
                         model.parameters())
    optimizer = torch.optim.Adam([{'params':base_params},{'params':pre_params, 'lr':args.pre_lr}],lr=args.lr)
    # optimizer = torch.optim.SGD([{'params':base_params},
    #                              {'params':pre_params, 'lr':args.pre_lr}],
    #                             lr=args.lr, momentum=0.9)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

mean_loss = 100
EPS = 1e-12
for epoch in range(args.epochs):
    print 'This model is %s_%s_%s'%(model_name,args.n_class,args.img_size)
    model.train()
    loss_list = []
    random.shuffle(img_list)

    if (epoch % 40 == 0) and epoch != 0:
        args.lr /= 10
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for i, path in enumerate(img_list):
        [ori_img, img_75, img_50], [ori_label, label_75, label_50], tmp_gt = get_multi_scale_data(args.data_path, path,
                                                                                                  img_size=args.img_size,
                                                                                                  n_classes=args.n_class)
        ori_out = model(ori_img)[0]
        ori_out = torch.log(softmax_2d(ori_out) + EPS)
        loss_1 = lossfunc(ori_out, ori_label)

        out_75 = model(img_75)[0]
        out_75 = torch.log(softmax_2d(out_75) + EPS)
        loss_2 = lossfunc(out_75, label_75)

        out_50 = model(img_50)[0]
        out_50 = torch.log(softmax_2d(out_50) + EPS)
        loss_3 = lossfunc(out_50, label_50)

        out_75 = F.upsample(out_75, size=(args.img_size, args.img_size), mode='bilinear')
        out_50 = F.upsample(out_50, size=(args.img_size, args.img_size), mode='bilinear')
        out = torch.max(ori_out, out_75)
        out = torch.max(out, out_50)
        loss = lossfunc(out, ori_label)

        loss += loss_1 + loss_2 + loss_3

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
            log_str = "%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (
            epoch, i, loss, meanIU, pixelAccuracy, meanAccuracy, classAccuracy)
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
            ROI_img = get_truth(ROI_img, path, [left, right], top)

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
