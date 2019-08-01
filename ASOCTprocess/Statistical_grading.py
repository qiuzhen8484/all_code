# -*- coding: utf-8 -*-
"""
Statical nucleus, lens, cortex information to establish grading indicator of the difficulty of segmentation
author: Shihao Zhang
Data: 2018/12/6
"""
import torch
import torch.nn as nn

import cv2
import numpy as np
import os
import pickle as pkl
import argparse
import time

import scipy.io as scio
import pandas as pd

import math


from core.utils import find_3_region,compute_not_csv_loss,get_img_list,get_data_16bit,get_model

from core.models import DeepGuidedFilter




# --------------------------------------------------------------------------------

models_list = ['UNet128', 'UNet256', 'UNet512', 'UNet1024', 'UNet256_kernel', 'UNet512_kernel', 'UNet1024_kernel', 'UNet256_kernel_dgf',
               'UNet512_SideOutput',  'UNet1024_SideOutput',
              'resnet_50', 'resnet_dense',  'PSP',       'dense_net',  'UNet128_deconv',      'UNet1024_deconv',
               'FPN18',    'FPN_deconv',    'My_UNet',    'M_Net',      'M_Net_deconv',       'Small',
               'VGG',      'Patch_Model',    'FED',       'finetune',    'MobileNet']
torch.manual_seed(1111)

# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')
# ---------------------------
# params do not need to change
# ---------------------------
parser.add_argument('--flag', type=str,  default='train',
                    help='decide to choice the dataset')
parser.add_argument('--results', type=str,  default='../data/visual_results',
                    help='path to save the visualization image')
parser.add_argument('--epochs', type=int, default=100,
                    help='the epochs of this run')
parser.add_argument('--n_class', type=int, default=4,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--pre_lr', type=float, default=0.000025,
                    help='pretrain model parameters learning rate, eg pretrain_resnet50')
parser.add_argument('--patch', type=bool, default=False,
                    help='trained on the patch')
parser.add_argument('--test_every_step', type=bool, default=False,
                    help='test after every train step')
parser.add_argument('--pre_part', type=bool, default=False,
                    help='pretrain the pytorch pretrain_model, eg resnet50 and resnet18, used in resnet_50 and resnet_dense')
parser.add_argument('--hard_example_train', type=bool, default=False,
                    help='only train the hard example')
parser.add_argument('--resize_img', type=bool, default=True,
                    help='decide to resize the input image')
parser.add_argument('--GroupNorm', type=bool, default=True,
                    help='decide to use the GroupNorm')
parser.add_argument('--BatchNorm', type=bool, default=False,
                    help='decide to use the BatchNorm')
parser.add_argument('--print_train', type=int, default=1,
                    help='interval to print information of trained')
parser.add_argument('--test_interval', type=int, default=10,
                    help='How many steps to test ')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay (L2 penalty) ')
parser.add_argument('--log_bad_data', type=bool, default=True,
                    help='decide to use the GroupNorm')
parser.add_argument('--init_param', type=bool, default=False,
                    help='init weight')
# ---------------------------
# model
# ---------------------------
parser.add_argument('--data_path', type=str, default='../data/dataset_16_LRS_final',
                    help='dir of the all img')
parser.add_argument('--best_model', type=str,  default='25_0.0001.pth',
                    help='the pretrain model')
parser.add_argument('--pre_all', type=bool, default=False,
                    help='pretrain the whole model')
parser.add_argument('--model_id', type=int, default=4,
                    help='the id of choice_model in models_list')
parser.add_argument('--batch_size', type=int, default=1,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=256,
                    help='the train img size')
parser.add_argument('--my_description', type=str, default='statical',
                    help='some description define your train')
# ---------------------------
# GPU
# ---------------------------
parser.add_argument('--use_gpu', type=bool, default=True,
                    help='dir of the all ori img')
parser.add_argument('--gpu_avaiable', type=str, default='6',
                    help='the Gpu used')
# ---------------------------
# guided_filter
# ---------------------------
parser.add_argument('--guided_filter', type=bool, default=False,
                    help='using guided_filter')
parser.add_argument('--nn_dgf_r', type=int, default=5, help='dgf radius')
parser.add_argument('--nn_dgf_eps', type=float, default=1e-1, help='dgf eps')
parser.add_argument('--nn_dgf_cn', type=int, default=15, help='adaptive layer number')
# ---------------------------
# optional
# ---------------------------
parser.add_argument('--use_crop', type=bool, default=False,
                    help='use the croied images which croied up and dowm')
parser.add_argument('--use_ployfit', type=bool, default=False,
                    help='use the croied images which croied up and dowm')
parser.add_argument('--health_flag', type=str,  default='all',
                    help='decide to choice the type of patient')

args = parser.parse_args()
print(args)

# --------------------------------------------------------------------------------

croped_up_and_down_dict={}
if args.use_crop:
    with open(os.path.join(args.data_path, 'croped_up_and_down_dict.pkl')) as f:
        croped_up_and_down_dict = pkl.load(f)
# --------------------------------------------------------------------------------


def fast_test(model, args, img_list, img_infor, model_name, save_test_img=False, log_bad_data=False,copy_up_down=False):
    path_my = r'../data/visual_lines/%s_%s_LGC' % (model_name, args.my_description)

    # -------------------------------
    # set list for statical information
    # -------------------------------

    health_list = ['T', 't', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    sick_list = ['C', 'M','c','S']
    uncertain_list = ['H', 'N', 'n', 'h']

    # list to store density information, set all the list to []

    file_name, h_type, lens, nucleus, cortex, lens_up, lens_down, \
    cortex_up, cortex_down, nucleus_up, nucleus_down = [[] for i in range(11)]


    softmax_2d = nn.Softmax2d()
    EPS = 1e-12
    ori_data = os.path.join(args.data_path, 'data')

    time_1_list = []
    time_2_list = []
    for i, path in enumerate(img_list):
        start = time.time()
        img, gt, tmp_gt, img_filter, gt_filter, tmp_gt_filter = get_data_16bit(args.data_path, [path], img_size=args.img_size, n_classes=args.n_class, gpu=args.use_gpu)
        model.eval()

        if 'M_Net' in model_name:
            out, side_5, side_6, side_7, side_8 = model(img)
            out = torch.log(softmax_2d(out) + EPS)
        elif 'dgf'in model_name:
            out = model(img, img_filter)[0]
            out = torch.log(softmax_2d(out) + EPS)
        else:
            out = model(img)[0]
            out = torch.log(softmax_2d(out) + EPS)

        # we resize image to shape of 1266*360 as default
        if args.guided_filter:
            ppi = np.argmax(out.cpu().data.numpy(), 1).reshape((1266, 360))
            # ppi = np.argmax(out.cpu().data.numpy(), 1).reshape((1024, 1024))
        else:
            # ppi = np.argmax(out.cpu().data.numpy(), 1).reshape((args.img_size, args.img_size))
            ppi = np.argmax(out.cpu().data.numpy(), 1).reshape((512, 64))

        loss_1, loss_2, loss_3, loss_NucleusBoundary_up, loss_NucleusBoundary_down, loss_LensBoundary_up, loss_LensBoundary_down, loss_LensBoundary2_up, loss_LensBoundary2_down = compute_not_csv_loss(ppi, path, args.data_path, args.use_crop, ployfit=args.use_ployfit)
        # loss_1, loss_2, loss_3 = compute_not_csv_loss_use_resize(ppi, path, args.data_path,args.img_size)

        file_name.append(path)
        if img_list[i][0] in health_list:
            h_type.append('health')
        elif img_list[i][0] in sick_list:
            h_type.append('sick')
        elif img_list[i][0] in uncertain_list:
            h_type.append('uncertain')
        else:
            print('images name error')

        nucleus.append(np.mean(loss_1))
        cortex.append(np.mean(loss_2))
        lens.append(np.mean(loss_3))
        nucleus_up.append(np.mean(loss_NucleusBoundary_up))
        nucleus_down.append(np.mean(loss_NucleusBoundary_down))
        cortex_up.append(np.mean(loss_LensBoundary_up))
        cortex_down.append(np.mean(loss_LensBoundary_down))
        lens_up.append(np.mean(loss_LensBoundary2_up))
        lens_down.append(np.mean(loss_LensBoundary2_down))

        start_1 = time.time()
        # if save_test_img:
        #     new_Image = ppi.astype(np.uint8)
        #
        #     tmp_out = new_Image.reshape([-1])
        #     tmp_gt = tmp_gt.reshape([-1])
        #
        #     confusion = np.zeros([args.n_class, args.n_class])
        #     for idx in range(len(tmp_gt)):
        #         confusion[tmp_gt[idx], tmp_out[idx]] += 1
        #
        #     ori_name = path[:-3]+'mat'
        #     left, right, top = img_infor[ori_name]
        #
        #     new_Image = ppi.astype(np.uint8)
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #     new_Image = cv2.morphologyEx(new_Image, cv2.MORPH_CLOSE, kernel)
        #
        #     ori_img_path = os.path.join(ori_data, ori_name)
        #     ori_img = scio.loadmat(ori_img_path)['mm']
        #     ori_img = np.stack((ori_img, ori_img, ori_img), axis=2)
        #     tmp_img = ori_img.copy()
        #     w= ori_img.shape[0] - top
        #     h = right - left
        #
        #     if args.use_crop:
        #         [h1,h2]=croped_up_and_down_dict[path]
        #         pred_img = cv2.resize(new_Image.astype(np.uint8), (h, h2 - h1),
        #                               interpolation=cv2.INTER_LANCZOS4)
        #     else:
        #         pred_img = cv2.resize(new_Image.astype(np.uint8), (h, w),
        #                               interpolation=cv2.INTER_LANCZOS4)
        #
        #
        #     tmp_mask_img = pred_img.copy()
        #
        #     NucleusBoundary, LensBoundary, LensBoundary2, NucleusBoundary_up, NucleusBoundary_down, LensBoundary_up, LensBoundary_down, LensBoundary2_up, LensBoundary2_down = find_3_region(tmp_mask_img,ployfit=args.use_ployfit)
        #
        #     if args.use_crop:
        #         tmp_img[top:top+w, left:right][h1:h2,:][NucleusBoundary]=[0,255,255]  # yellow
        #         tmp_img[top:top+w, left:right][h1:h2,:][LensBoundary] = [220, 20, 60]  # crimson
        #         tmp_img[top:top+w, left:right][h1:h2,:][LensBoundary2] = [255, 0, 255]  # fuchsia
        #     else:
        #         tmp_img[top:top+w, left:right][NucleusBoundary]=[0,255,255]  # yellow
        #         tmp_img[top:top+w, left:right][LensBoundary] = [0,255,0]
        #         tmp_img[top:top+w, left:right][LensBoundary2] = [255,255,0]
        #
        #     if not os.path.exists(path_my):
        #         os.mkdir(path_my)
        #     cv2.imwrite(os.path.join(path_my, '%s.png' % path[:-4]), tmp_img)

        end = time.time()
        time_2_list.append(end-start)
        time_1_list.append(start_1-start)
        print(str(i+1)+r'/'+str(len(img_list))+': '+'Nuclear: %s, Cortex: %s, Lens: %s, time1: %s, time2: %s' % (str(np.mean(loss_1)), str(np.mean(loss_2)), str(np.mean(loss_3)), end-start, start_1-start))

    data_all = {'file_name': file_name, 'h_type': h_type, 'lens': lens,
                'cortex': cortex, 'nucleus': nucleus,
                'lens_up': lens_up,  'lens_down': lens_down,
                'cortex_up': cortex_up, 'cortex_down': cortex_down,
                'nucleus_up': nucleus_up, 'nucleus_down': nucleus_down}

    df = pd.DataFrame(data_all)
    print(df.describe())
    df.describe().to_csv("describe.csv")
    df.to_csv("data.csv")








if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable

    model_name = models_list[args.model_id]
    # model = get_cur_model(model_name, n_classes=args.n_class, bn=args.GroupNorm, pretrain=args.pre_part)

    model = get_model(model_name)
    if 'dgf' in model_name:
        model = model(n_classes=args.n_class, bn=args.GroupNorm, radius=args.nn_dgf_r, eps=args.nn_dgf_eps,cn=args.nn_dgf_cn)
    else:
        model = model(n_classes=args.n_class, bn=args.GroupNorm)
    # model = nn.DataParallel(model)

    if args.use_gpu:
        model.cuda()
    if True:
        model_path = '/home/intern1/zhangshihao/project/ASOCT-new/models/UNet256_kernel_LRS_final_new/25_0.0001.pth'
        model.load_state_dict(torch.load(model_path))
        print('success load models: %s_%s' % (model_name, args.my_description))

    print('This model is %s_%s_%s' % (model_name, args.n_class, args.img_size))
    # model=model.module

    img_infor, img_list = get_img_list(args.data_path, flag='train', need_infor=True, health_flag=args.health_flag) #['train','val','hard','mix']

    # img_list = img_list[5913:5915]
    # img_list_tmp = img_list[0:5913] + img_list[5915:9379]+ img_list[9380:]
    # img_list_tmp = img_list[0:9379] + img_list[9380:]
    # img_list = img_list_tmp

    # img_list = []
    # img_infor=[]
    # with open('LBO_gooddata.txt') as fin:
    #     for line in fin:
    #         img_list.append(line.strip('\n'))

    fast_test(model, args, img_list,img_infor, model_name, save_test_img=False, log_bad_data=False)

