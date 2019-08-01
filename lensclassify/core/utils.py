from __future__ import division
import numpy as np
import pandas as pd
import os
import cv2
import torch
import pickle as pkl
import random
import math
from skimage import segmentation
from models import *
from models_classification import UNet256_kernel_classification
from torchvision import models
from torchvision.models import resnet50
from ImageData import ImageData
import scipy.io as scio
import time

from fpn import FPN18
import fcn
import pspnet
import deeplab_resnet
from skimage import exposure
from ModelUNetTogether import UNet
# import ConfigParser
class Read_ini(object):

    def __init__(self,path):
        super(Read_ini, self).__init__()
        self.conf = ConfigParser.ConfigParser()
        self.conf.read(path)

    def write(self, new_option, key, value):
        try:
            self.conf.add_section(new_option)
        except:
            print('%s already exits'%new_option)
        self.conf.set(new_optine, key, value)
        self.conf.write(open(path, 'w'))

    def read(self, new_option, key, type=1):
        """
        :param new_option:
        :param key:
        :param type: [1]float [2]int [3]str
        :return:
        """
        if type==1:
            return float(self.conf.get(new_option, key))
        elif type==2:
            return int(self.conf.get(new_option, key))
        else:
            return self.conf.get(new_option, key)


def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I


def random_shift_scale_rotateN(images, shift_limit=(-0.0625,0.0625), scale_limit=(1/1.1,1.1),
                               rotate_limit=(-45,45), aspect_limit = (1,1),  borderMode=cv2.BORDER_REFLECT_101 , u=0.5):
    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    if random.random() < u:
        height,width,channel = images[0].shape

        angle  = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
        scale  = random.uniform(scale_limit[0],scale_limit[1])
        aspect = random.uniform(aspect_limit[0],aspect_limit[1])
        sx    = scale*aspect/(aspect**0.5)
        sy    = scale       /(aspect**0.5)
        dx    = round(random.uniform(shift_limit[0],shift_limit[1])*width )
        dy    = round(random.uniform(shift_limit[0],shift_limit[1])*height)

        cc = math.cos(angle/180*math.pi)*(sx)
        ss = math.sin(angle/180*math.pi)*(sy)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)

        for n, image in enumerate(images):
            images[n] = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return images

# def get_mini_batch_data(data_path, img_name, img_size=256,gpu_num=1):
#
#     img_path = os.path.join(data_path, 'train_data', img_name)
#     label_path = os.path.join(data_path, 'train_label', img_name)
#
#     img = cv2.imread(img_path)
#     label = cv2.imread(label_path)
#     img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
#     label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]
#
#     # Data Augmentation:
#
#     img, label = random_shift_scale_rotateN([img,label])
#
#     img = np.transpose(img, [2, 0, 1])
#
#     tmp_gt = label.copy()
#     label = np.transpose(label, [2, 0, 1])
#
#     img = Variable(torch.from_numpy(img)).float().cuda(gpu_num)
#     img = torch.unsqueeze(img, 0)
#     label = Variable(torch.from_numpy(label)).long().cuda(gpu_num)
#
#     return img, label,tmp_gt


# def get_new_data(data_path, img_name, img_size=256, n_classes=4):
#
#     def get_label(label):
#         tmp_gt = label.copy()
#         label = np.transpose(label, [2, 0, 1])
#         label = Variable(torch.from_numpy(label)).long().cuda()
#         return label,tmp_gt
#
#     if n_classes==2:
#         img_path = os.path.join(data_path,str(n_classes), 'train_data', img_name)
#         label_path = os.path.join(data_path,str(n_classes), 'train_label', img_name)
#
#         img = cv2.imread(img_path)
#         label = cv2.imread(label_path)
#         img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
#         label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]
#         img = np.transpose(img, [2, 0, 1])
#         img = Variable(torch.from_numpy(img)).float().cuda()
#         img = torch.unsqueeze(img, 0)
#         label, tmp_gt = get_label(label)
#         label_list = [[label, tmp_gt]]
#         return img, label_list
#
#     img_path = os.path.join(data_path,str(n_classes), 'train_data', img_name)
#     label_path = os.path.join(data_path, str(n_classes),'train_label', img_name)
#
#     img = cv2.imread(img_path)
#     label = cv2.imread(label_path)
#     img, label = data_arguementaion(img, label)
#     img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
#     label_1 = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]
#     label_2 = cv2.resize(label, (img_size/2, img_size/2), interpolation=cv2.INTER_AREA)[:, :, :1]
#     label_3 = cv2.resize(label, (img_size/4, img_size/4), interpolation=cv2.INTER_AREA)[:, :, :1]
#     label_4 = cv2.resize(label, (img_size/8, img_size/8), interpolation=cv2.INTER_AREA)[:, :, :1]
#     label_5 = cv2.resize(label, (img_size/16, img_size/16), interpolation=cv2.INTER_AREA)[:, :, :1]
#     label_6 = cv2.resize(label, (img_size / 32, img_size / 32), interpolation=cv2.INTER_AREA)[:, :, :1]
#
#     img = np.transpose(img, [2, 0, 1])
#     img = Variable(torch.from_numpy(img)).float().cuda()
#     img = torch.unsqueeze(img, 0)
#
#     label_1, tmp_gt_1 = get_label(label_1)
#     label_2, tmp_gt_2 = get_label(label_2)
#     label_3, tmp_gt_3 = get_label(label_3)
#     label_4, tmp_gt_4 = get_label(label_4)
#     label_5, tmp_gt_5 = get_label(label_5)
#     label_6, tmp_gt_6 = get_label(label_6)
#
#     label_list = [[label_1, tmp_gt_1], [label_2, tmp_gt_2], [label_3, tmp_gt_3], [label_4, tmp_gt_4],
#                   [label_5, tmp_gt_5], [label_6, tmp_gt_6]]
#
#     return img, label_list

def get_data(data_path, img_name, img_size=256, n_classes=4, gpu=True, just_img=False):

    def get_label(label):
        tmp_gt = label.copy()
        # label = np.transpose(label, [2, 0, 1])
        label = label.astype(np.int64)
        label = Variable(torch.from_numpy(label)).long()
        if gpu:
            label = label.cuda()
        return label,tmp_gt

    images = []
    labels = []
    tmp_gts = []
    batch_size = len(img_name)
    for i in range(batch_size):

        img_path = os.path.join(data_path, 'train_data', img_name[i])
        label_path = os.path.join(data_path,'train_label', img_name[i])
        # print img_path

        img = cv2.imread(img_path)
        label = cv2.imread(label_path, 0)
        # img, label = data_arguementaion(img, label)

        if n_classes == 2:
            new_label = np.zeros_like(label)
            new_label[label == 3] = 1
            label = new_label
        img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)

        img = np.transpose(img, [2, 0, 1])
        img = Variable(torch.from_numpy(img)).float()
        if gpu:
            img = img.cuda()
        label, tmp_gt = get_label(label)
        images.append(img)
        labels.append(label)
        tmp_gts.append(tmp_gt)
    images = torch.stack(images)
    if just_img:
        return images # improve the domo speed
    labels = torch.stack(labels)
    tmp_gts = np.stack(tmp_gts)

    return images, labels, tmp_gts


def get_data_guided(data_path, img_name, img_size=256, n_classes=4, gpu=True, just_img=False,use_crop=False,croped_up_and_down_dict={}):

    def get_label(label):
        tmp_gt = label.copy()
        # label = np.transpose(label, [2, 0, 1])
        label = label.astype(np.int64)
        label = Variable(torch.from_numpy(label)).long()
        if gpu:
            label = label.cuda()
        return label,tmp_gt

    images = []
    labels = []
    tmp_gts = []

    images_filter = []
    labels_filter = []
    tmp_gts_filter = []
    batch_size = len(img_name)
    for i in range(batch_size):

        img_path = os.path.join(data_path, 'train_data', img_name[i])
        label_path = os.path.join(data_path,'train_label', img_name[i])
        # print img_path

        img = cv2.imread(img_path)
        # if True means using equalizeHist
        # if True:
        #     img = cv2.equalizeHist(img)
        label = cv2.imread(label_path, 0)
        if use_crop:
            [h1,h2]=croped_up_and_down_dict[img_name[i]]
            label=label[h1:h2,:]

        # img, label = data_arguementaion(img, label)

        if n_classes == 2:
            new_label = np.zeros_like(label)
            new_label[label == 3] = 1
            label = new_label

        img_filter = cv2.resize(img, (950, 1256),interpolation=cv2.INTER_AREA)
        label_filter = cv2.resize(label, (950, 1256), interpolation=cv2.INTER_AREA)

        # img_filter = cv2.resize(img, (1024, 1024),interpolation=cv2.INTER_AREA)
        # label_filter = cv2.resize(label, (1024, 1024), interpolation=cv2.INTER_AREA)

        img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)

        img = np.transpose(img, [2, 0, 1])
        img = Variable(torch.from_numpy(img)).float()

        img_filter = np.transpose(img_filter, [2, 0, 1])
        img_filter = Variable(torch.from_numpy(img_filter)).float()

        if gpu:
            img = img.cuda()
            img_filter=img_filter.cuda()

        label, tmp_gt = get_label(label)
        label_filter, tmp_gt_filter = get_label(label_filter)

        images.append(img)
        labels.append(label)
        tmp_gts.append(tmp_gt)

        images_filter.append(img_filter)
        labels_filter.append(label_filter)
        tmp_gts_filter.append(tmp_gt_filter)

    images = torch.stack(images)

    images_filter = torch.stack(images_filter)

    if just_img:
        return images # improve the domo speed
    labels = torch.stack(labels)
    tmp_gts = np.stack(tmp_gts)

    labels_filter = torch.stack(labels_filter)
    tmp_gts_filter = np.stack(tmp_gts_filter)

    return images, labels, tmp_gts, images_filter, labels_filter, tmp_gts_filter


def get_data_16bit(data_path, img_name, img_size=256, n_classes=4, gpu=True, just_img=False,unbalanced=True):
    """
    Load 16bit data
    Author: Shihao Zhang
    :param data_path:
    :param img_name:
    :param img_size:
    :param n_classes:
    :param gpu:
    :param just_img:
    :param unbalanced:
    :return:
    """

    def get_label(label):
        tmp_gt = label.copy()
        # label = np.transpose(label, [2, 0, 1])
        label = label.astype(np.int64)
        label = Variable(torch.from_numpy(label)).long()
        if gpu:
            label = label.cuda()
        return label,tmp_gt

    images = []
    labels = []
    tmp_gts = []

    images_filter = []
    labels_filter = []
    tmp_gts_filter = []
    batch_size = len(img_name)
    for i in range(batch_size):

        img_path = os.path.join(data_path, 'train_data', img_name[i][:-4] + '.mat')
        label_path = os.path.join(data_path,'train_label', img_name[i][:-4] + '.png')

        img = scio.loadmat(img_path)['mm']
        # if img.shape[2]>1:
        #     img = img[:,:,0]
        #     img = img[:,:,np.newaxis]
        img = img[:, :, np.newaxis]
        label = cv2.imread(label_path, 0)

        if n_classes == 2:
            new_label = np.zeros_like(label)
            new_label[label == 3] = 1
            label = new_label

        # Shape of the 16 bit data is different from 8 bit,
        #  so we resize the filter data to shape W_H: 360_1266.
        if unbalanced :
            img_filter = cv2.resize(img, (360, 1266),interpolation=cv2.INTER_AREA)
            label_filter = cv2.resize(label, (360, 1266), interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (64, 512),interpolation=cv2.INTER_AREA)
            label = cv2.resize(label, (64, 512), interpolation=cv2.INTER_AREA)

        else:
            img_filter = cv2.resize(img, (1024, 1024),interpolation=cv2.INTER_AREA)
            label_filter = cv2.resize(label, (1024, 1024), interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
            label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)


        img = np.stack((img, img, img), axis=2)
        img_filter = np.stack((img_filter, img_filter, img_filter), axis=2)
        img = np.int32(img)
        img_filter = np.int32(img_filter)

        img = np.transpose(img, [2, 0, 1])
        img = Variable(torch.from_numpy(img)).float()

        img_filter = np.transpose(img_filter, [2, 0, 1])
        img_filter = Variable(torch.from_numpy(img_filter)).float()

        if gpu:
            img = img.cuda()
            img_filter=img_filter.cuda()

        label, tmp_gt = get_label(label)
        label_filter, tmp_gt_filter = get_label(label_filter)

        images.append(img)
        labels.append(label)
        tmp_gts.append(tmp_gt)

        images_filter.append(img_filter)
        labels_filter.append(label_filter)
        tmp_gts_filter.append(tmp_gt_filter)

    images = torch.stack(images)

    images_filter = torch.stack(images_filter)

    if just_img:
        return images # improve the domo speed
    labels = torch.stack(labels)
    tmp_gts = np.stack(tmp_gts)

    labels_filter = torch.stack(labels_filter)
    tmp_gts_filter = np.stack(tmp_gts_filter)

    return images, labels, tmp_gts, images_filter, labels_filter, tmp_gts_filter


def get_data_nolabel(data_path, img_name, img_size=256, n_classes=4, gpu=True, flag_16bit=True,unbalanced=True):

    images = []
    images_filter = []
    top_list=[]
    left_list=[]
    right_list=[]

    imageData=ImageData()

    batch_size = len(img_name)
    for i in range(batch_size):

        img_path = os.path.join(data_path, 'data', img_name[i])

        if flag_16bit:
            img = scio.loadmat(img_path)['mm']
            img = np.stack([img, img, img], axis=2)
        else:
            img = cv2.imread(img_path)

        img_shape = img.shape
        top = int(img_shape[1] / 3 - 100)

        if flag_16bit:
            top = int(600)
            img_tmp = scio.loadmat(img_path)['mm']
            img_tmp = np.squeeze(img_tmp)
        else:
            img_tmp = cv2.imread(img_path, 0)

        img_tmp = cv2.medianBlur(img_tmp, 5)

        if flag_16bit:
            left, right = get_left_right(img_tmp)
        else:
            left, right = imageData.GetLeftAndRight(img_tmp)
        if right <= left:
            right= left+50

        # if flag_16bit:
        #     right = right +100
        #     left = left -150
        # # img = img[top:, left:right, :]

        img = img[top:,left:right, :]


        if unbalanced :
            img_filter = cv2.resize(img, (360, 1266),interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (64, 512),interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            img_filter = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)


        if flag_16bit:
            img = np.int32(img)
            img_filter = np.int32(img_filter)

        img = np.transpose(img, [2, 0, 1])
        img = Variable(torch.from_numpy(img)).float()

        img_filter = np.transpose(img_filter, [2, 0, 1])
        img_filter = Variable(torch.from_numpy(img_filter)).float()

        if gpu:
            img = img.cuda()
            img_filter=img_filter.cuda()

        images.append(img)
        images_filter.append(img_filter)
        top_list.append(top)
        left_list.append(left)
        right_list.append(right)

    images = torch.stack(images)
    images_filter = torch.stack(images_filter)

    return images, images_filter,top_list,left_list,right_list


def get_data_with_given_label(data_path, img_name, img_size=256, n_classes=4, gpu=True, flag_16bit=True,unbalanced=True):

    images = []
    images_filter = []
    top_list=[]
    left_list=[]
    right_list=[]
    down_list= []

    imageData=ImageData()

    batch_size = len(img_name)
    for i in range(batch_size):

        img_path = os.path.join(data_path, 'data', img_name[i])

        img = cv2.imread(img_path)
        img_shape = img.shape

        mat_path = img_path[:-4]+'_1.mat'
        down = scio.loadmat(mat_path)['bottom'][0][0]
        top = scio.loadmat(mat_path)['top'][0][0]
        left = scio.loadmat(mat_path)['left'][0][0]
        right = scio.loadmat(mat_path)['right'][0][0]

        img = img[top:down,left:right, :]


        if unbalanced :
            img_filter = cv2.resize(img, (360, 1266),interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (64, 512),interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            img_filter = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)


        if flag_16bit:
            img = np.int32(img)
            img_filter = np.int32(img_filter)

        img = np.transpose(img, [2, 0, 1])
        img = Variable(torch.from_numpy(img)).float()

        img_filter = np.transpose(img_filter, [2, 0, 1])
        img_filter = Variable(torch.from_numpy(img_filter)).float()

        if gpu:
            img = img.cuda()
            img_filter=img_filter.cuda()

        images.append(img)
        images_filter.append(img_filter)
        top_list.append(top)
        left_list.append(left)
        right_list.append(right)
        down_list.append(down)

    images = torch.stack(images)
    images_filter = torch.stack(images_filter)

    return images, images_filter,top_list,left_list,right_list,down_list

def get_data_nolabel_updown(data_path, img_name, img_size=256, n_classes=4, gpu=True, flag_16bit=True,unbalanced=True):

    images = []
    images_filter = []
    top_list=[]
    left_list=[]
    right_list=[]
    down_list= []

    imageData=ImageData()

    batch_size = len(img_name)
    for i in range(batch_size):

        img_path = os.path.join(data_path, 'data', img_name[i])

        if flag_16bit:
            img = scio.loadmat(img_path)['mm']
            img = np.stack([img, img, img], axis=2)
        else:
            img = cv2.imread(img_path)

        img_shape = img.shape

        if flag_16bit:
            img_tmp = scio.loadmat(img_path)['mm']
            img_tmp = np.squeeze(img_tmp)
        else:
            img_tmp = cv2.imread(img_path, 0)


        tmp_path = img_path[:-4]+'_1.mat'
        top = scio.loadmat(tmp_path)['top'][0,0]
        left = scio.loadmat(tmp_path)['left'][0,0]
        right = scio.loadmat(tmp_path)['right'][0, 0]
        down = scio.loadmat(tmp_path)['bottom'][0, 0]

        # if flag_16bit:
        #     right = right +100
        #     left = left -150
        # # img = img[top:, left:right, :]

        img = img[top:down,left:right, :]


        if unbalanced :
            img_filter = cv2.resize(img, (360, 1266),interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (64, 512),interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            img_filter = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)


        if flag_16bit:
            img = np.int32(img)
            img_filter = np.int32(img_filter)

        img = np.transpose(img, [2, 0, 1])
        img = Variable(torch.from_numpy(img)).float()

        img_filter = np.transpose(img_filter, [2, 0, 1])
        img_filter = Variable(torch.from_numpy(img_filter)).float()

        if gpu:
            img = img.cuda()
            img_filter=img_filter.cuda()

        images.append(img)
        images_filter.append(img_filter)
        top_list.append(top)
        left_list.append(left)
        right_list.append(right)
        down_list.append(down)

    images = torch.stack(images)
    images_filter = torch.stack(images_filter)

    return images, images_filter,top_list,left_list,right_list,down_list


def get_left_right(img):
    """
    This function aims to get the left and right boundaries of AS-OCT figure
    Author: Shihao Zhang
    :param img: The AS-OCT figure
    :return: left boundary, right boundary
    """
    t1 = time.time()
    shape = img.shape
    height = shape[0]
    width = shape[1]

    height_3_4 = height*3//4
    height_9_10 = height*9//10
    pixel = img[(height//2)+100:height, 10:30]
    pixel_mean = np.mean(pixel)
    high = pixel_mean * 1.010
    high1 = pixel_mean * 1.020
    tmp = np.arange(width)
    left = 0
    right = 0
    for i in tmp[20:]:
        pixel_local = img[height_3_4-100:height_9_10-50, i-5:i]
        pixel_local_mean = np.mean(pixel_local)
        if pixel_local_mean>high:
            if i + 20 > width:
                print('boundary error')
                continue
            pixel_local_a = img[height_3_4 - 100:height_9_10-50, i - 5+20:i+20]
            pixel_local_mean_a = np.mean(pixel_local_a)
            if pixel_local_mean_a > high1:
                left = i-2
                break

    tmp = tmp[::-1]  # reverse the array to find the right boundary
    for i in tmp[20:]:
        pixel_local = img[height_3_4-100:height_9_10-50, i:i+5]
        pixel_local_mean = np.mean(pixel_local)
        if pixel_local_mean>high:
            if i-20 < 0:
                print('boundary error')
                continue
            pixel_local_a = img[height_3_4 - 100:height_9_10-50, i-20:i+5-20]
            pixel_local_mean_a = np.mean(pixel_local_a)
            if pixel_local_mean_a > high1:
                right = i+3
                break


    t2 = time.time()
    print("get left and right position time:  " + str(t2 - t1))

    return left, right


def get_left_right_new(img):
    """
    This function aims to get the left and right boundaries of AS-OCT figure
    Author: Shihao Zhang
    :param img: The AS-OCT figure
    :return: left boundary, right boundary
    """
    t1 = time.time()
    shape = img.shape

    a1 = 65535 / 256 * 150
    a2 = 65535 / 256 * 84
    img[img<a2]=0
    img[img>a1]=255
    temp1 = (img<a1) +(img>a2)
    img[temp1] = (img[temp1] -a2/(a1-a2))*255
    threshold  = 300

    height = shape[0]
    width = shape[1]


    tmp = np.arange(width)
    left = 0
    right = 0
    for i in tmp[22:]:
        pixel_local_b = img[600:,i-2]
        pixel_local = img[600:, i - 1]
        pixel_local_a = img[600:, i ]
        if (np.sum(pixel_local>10)-np.sum(pixel_local_b>10))>threshold:
            if (np.sum(pixel_local_a>10)-np.sum(pixel_local>10))<threshold:
                left = i-1
                break

    tmp = tmp[::-1]  # reverse the array to find the right boundary
    for i in tmp[22:]:
        pixel_local_b = img[600:,i-2]
        pixel_local = img[600:, i - 1]
        pixel_local_a = img[600:, i ]
        if (np.sum(pixel_local>10)-np.sum(pixel_local_b>10))>threshold:
            if (np.sum(pixel_local_a>10)-np.sum(pixel_local>10))<threshold:
                left = i-1
                break

    t2 = time.time()
    print("get left and right position time:  " + str(t2 - t1))

    return left, right


def get_data_classification(data_path, img_name, img_size=256, n_classes=4, gpu=True, just_img=False,use_crop=False,croped_up_and_down_dict={}):

    health_list=['T','t','0','1','2','3','4','5','6','7','8','9']
    sick_list=['C','M']
    uncertain_list=['H','N']

    def get_label(label):
        tmp_gt = label.copy()
        # label = np.transpose(label, [2, 0, 1])
        label = label.astype(np.int64)
        label = Variable(torch.from_numpy(label)).long()
        if gpu:
            label = label.cuda()
        return label,tmp_gt

    images = []
    labels = []
    tmp_gts = []
    labels_classification=[]

    images_filter = []
    labels_filter = []
    tmp_gts_filter = []
    batch_size = len(img_name)
    for i in range(batch_size):

        img_path = os.path.join(data_path, 'train_data', img_name[i])
        label_path = os.path.join(data_path,'train_label', img_name[i])
        # print img_path

        img = cv2.imread(img_path)
        # if True means using equalizeHist
        # if True:
        #     img = cv2.equalizeHist(img)
        label = cv2.imread(label_path, 0)
        if use_crop:
            [h1,h2]=croped_up_and_down_dict[img_name[i]]
            label=label[h1:h2, :]

        # img, label = data_arguementaion(img, label)

        # ------------------------------
        # labels for classification
        # 1 means health, and 0 means sick
        # ------------------------------
        label_classification = -1
        if img_name[i][0] in health_list:
            label_classification = 1
        elif img_name[i][0] in sick_list:
            label_classification = 0
        else:
            print('images name error')

        labels_classification.append(label_classification)

        if n_classes == 2:
            new_label = np.zeros_like(label)
            new_label[label == 3] = 1
            label = new_label
        img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)

        img_filter = cv2.resize(img, (1024, 1024),interpolation=cv2.INTER_AREA)
        label_filter = cv2.resize(label, (1024, 1024), interpolation=cv2.INTER_AREA)

        img = np.transpose(img, [2, 0, 1])
        img = Variable(torch.from_numpy(img)).float()

        img_filter = np.transpose(img_filter, [2, 0, 1])
        img_filter = Variable(torch.from_numpy(img_filter)).float()

        if gpu:
            img = img.cuda()
            img_filter=img_filter.cuda()

        label, tmp_gt = get_label(label)
        label_filter, tmp_gt_filter = get_label(label_filter)

        images.append(img)
        labels.append(label)
        tmp_gts.append(tmp_gt)

        images_filter.append(img_filter)
        labels_filter.append(label_filter)
        tmp_gts_filter.append(tmp_gt_filter)

    images = torch.stack(images)

    images_filter = torch.stack(images_filter)

    if just_img:
        return images # improve the domo speed
    labels = torch.stack(labels)
    tmp_gts = np.stack(tmp_gts)

    labels_filter = torch.stack(labels_filter)
    tmp_gts_filter = np.stack(tmp_gts_filter)

    labels_classification = Variable(torch.LongTensor(np.array(labels_classification))).cuda()

    return images, labels, tmp_gts, images_filter, labels_filter, tmp_gts_filter,labels_classification



def get_img_list(data_path, flag='train',health_flag='all', need_infor=False):
    with open(os.path.join(data_path, 'train_dict.pkl'), 'rb') as f:
        train_dict = pkl.load(f)
    if flag=='train':
        img_list = train_dict['test_list']
    elif flag=='val':
        img_list = train_dict['val_list']
    elif flag=='hard':
        with open('./logs/hard_example.pkl') as f:
            img_list = pkl.load(f)
    else:
        img_list = train_dict['val_list']+train_dict['train_list']

    health_list=['T','t','0','1','2','3','4','5','6','7','8','9']
    sick_list=['C','M']
    uncertain_list=['H','N']

    if health_flag=='health':
        img_list_temp=[]
        for picture_name in img_list:
            if (picture_name[0] in sick_list) or (picture_name[0] in uncertain_list):
                continue
            else:
                img_list_temp.append(picture_name)
        img_list=img_list_temp

    elif health_flag=='sick':
        img_list_temp = []
        for picture_name in img_list:
            if (picture_name[0] in health_list) or (picture_name[0] in uncertain_list):
                continue
            else:
                img_list_temp.append(picture_name)
            img_list = img_list_temp

    # for classification, contain sick and health images
    elif health_flag=='classification':
        img_list_temp = []
        for picture_name in img_list:
            if picture_name[0] in uncertain_list:
                continue
            else:
                img_list_temp.append(picture_name)
            img_list = img_list_temp

    if need_infor:
        with open(os.path.join(data_path, 'train.pkl'), 'rb') as f:
            img_infor = pkl.load(f)
        return img_infor, img_list
    return img_list



def get_img_val(data_path, flag='train',health_flag='all', need_infor=False):
    with open(os.path.join(data_path, 'train_dict_val.pkl')) as f:
        train_dict = pkl.load(f)
    if flag=='train':
        img_list = train_dict['train_list']
    elif flag=='val':
        img_list = train_dict['val_list']
    elif flag=='test':
        img_list = train_dict['test_list']
    else:
        print('train_test_val type error')

    health_list=['T','t','0','1','2','3','4','5','6','7','8','9']
    sick_list=['C','M']
    uncertain_list=['H','N']

    if health_flag=='health':
        img_list_temp=[]
        for picture_name in img_list:
            if (picture_name[0] in sick_list) or (picture_name[0] in uncertain_list):
                continue
            else:
                img_list_temp.append(picture_name)
        img_list=img_list_temp

    elif health_flag=='sick':
        img_list_temp = []
        for picture_name in img_list:
            if (picture_name[0] in health_list) or (picture_name[0] in uncertain_list):
                continue
            else:
                img_list_temp.append(picture_name)
            img_list = img_list_temp

    # for classification, contain sick and health images
    elif health_flag=='classification':
        img_list_temp = []
        for picture_name in img_list:
            if picture_name[0] in uncertain_list:
                continue
            else:
                img_list_temp.append(picture_name)
            img_list = img_list_temp

    if need_infor:
        with open(os.path.join(data_path, 'train.pkl')) as f:
            img_infor = pkl.load(f)
        return img_infor, img_list
    return img_list

# def get_ori_data(data_path, img_name, img_size=256, n_classes=4):
#
#     def get_label(label):
#         tmp_gt = label.copy()
#         label = np.transpose(label, [2, 0, 1])
#         # label = Variable(torch.from_numpy(label)).cuda()
#         label = Variable(torch.from_numpy(label)).long().cuda()
#         return label,tmp_gt
#
#     img_path = os.path.join(data_path,str(n_classes), 'train_data', img_name)
#     label_path = os.path.join(data_path, str(n_classes),'train_label', img_name)
#
#     img = cv2.imread(img_path)
#     # print img_path
#     label = cv2.imread(label_path)
#     # new_label = np.zeros_like(label)
#     # new_label[label==3]=1
#     # boundry = segmentation.find_boundaries(new_label)
#     # boundry = boundry*1
#     #
#     # new_label[new_label == 3] = 2
#     img, label = data_arguementaion(img, label)
#     img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
#     label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]
#     new_label = label.copy()
#     img = reconstruct_img(new_label, img, img_size,random=np.random.randint(35,45))
#     # cv2.imwrite('./logs/test_data/train.png',img)
#
#     img = np.transpose(img, [2, 0, 1])
#     img = Variable(torch.from_numpy(img)).float().cuda()
#     img = torch.unsqueeze(img, 0)
#
#     label, tmp_gt = get_label(label)
#
#     return img, label, tmp_gt
#
#
# def get_test_data(data_path, img_name, img_size=256, n_classes=4):
#
#     def get_label(label):
#         tmp_gt = label.copy()
#         label = np.transpose(label, [2, 0, 1])
#         label = Variable(torch.from_numpy(label)).long().cuda()
#         return label,tmp_gt
#
#     if n_classes==2:
#         img_path = os.path.join(data_path,str(n_classes), 'test_data', img_name)
#         label_path = os.path.join(data_path,str(n_classes), 'train_label', img_name)
#
#         img = cv2.imread(img_path)
#         label = cv2.imread(label_path)
#         label[label > 0] = 1
#         img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
#         label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]
#         img = np.transpose(img, [2, 0, 1])
#         img = Variable(torch.from_numpy(img)).float().cuda()
#         img = torch.unsqueeze(img, 0)
#         label, tmp_gt = get_label(label)
#         return img, label, tmp_gt
#
#     img_path = os.path.join(data_path,str(n_classes), 'test_data', img_name)
#     label_path = os.path.join(data_path, str(n_classes),'train_label', img_name)
#
#     img = cv2.imread(img_path)
#     label = cv2.imread(label_path)
#     # img, label = data_arguementaion(img, label)
#     img = cv2.resize(img, (img_size, img_size),interpolation=cv2.INTER_AREA)
#     label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]
#
#     img = np.transpose(img, [2, 0, 1])
#     img = Variable(torch.from_numpy(img)).float().cuda()
#     img = torch.unsqueeze(img, 0)
#
#     label, tmp_gt = get_label(label)
#
#     return img, label, tmp_gt
#
# def get_multi_data(data_path, target_path, position, img_size=256, n_classes=4):
#     left, right, top = position
#     def get_label(label):
#         tmp_gt = label.copy()
#         label = np.transpose(label, [2, 0, 1])
#         label = Variable(torch.from_numpy(label)).long().cuda()
#         return label,tmp_gt
#
#     # get the label
#     label_path = os.path.join(data_path, str(n_classes), 'train_label', target_path)
#     label = cv2.imread(label_path)
#     label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]
#     label, tmp_gt = get_label(label)
#
#     # get the multi_model concat img
#     left_right = {'L': '1', 'R': '0'}
#     target_infor = target_path.split('_')
#     img_id = target_infor[2]
#     copy_path = target_infor[0]+'_'+left_right[target_infor[1]]+'_'+target_infor[-1].split('.')[0]
#     # print copy_path
#     start_idx = 9+int(img_id)*5
#     end_idx = start_idx + 2
#     end_idx_1 = end_idx+1
#     target_img = []
#     for i in [start_idx, end_idx, end_idx_1]:
#         img_path = os.path.join(data_path, 'copy_data',copy_path,str(i)+'.jpg')
#         # print img_path
#         tmp_img = cv2.imread(img_path)
#         # print tmp_img.shape
#         tmp_img = tmp_img[top:,left:right]
#         tmp_img = cv2.resize(tmp_img, (img_size*2, img_size*2), interpolation=cv2.INTER_AREA)
#         tmp_img = np.transpose(tmp_img, [2, 0, 1])
#         tmp_img = Variable(torch.from_numpy(tmp_img)).float().cuda()
#         tmp_img = torch.unsqueeze(tmp_img, 0)
#         target_img.append(tmp_img)
#     return target_img,label,tmp_gt
#
# def get_patch_data(data_path, img_name, img_size=256, n_classes=4):
#
#     def get_label(label):
#         tmp_gt = label.copy()
#         label = np.transpose(label.copy(), [2, 0, 1])
#         label = torch.from_numpy(label)
#         label = Variable(label)
#         label = label.long()
#         label = label.cuda()
#         # label = Variable(torch.from_numpy(label)).long().cuda()
#         return label,tmp_gt
#
#     if n_classes==2:
#         img_path = os.path.join(data_path,str(n_classes), 'train_data', img_name)
#         label_path = os.path.join(data_path,str(n_classes), 'train_label', img_name)
#
#         img = cv2.imread(img_path)
#         label = cv2.imread(label_path)
#         img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
#         label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]
#         img = np.transpose(img, [2, 0, 1])
#         img = Variable(torch.from_numpy(img)).float().cuda()
#         img = torch.unsqueeze(img, 0)
#         label, tmp_gt = get_label(label)
#         return img, label, tmp_gt
#
#     img_path = os.path.join(data_path,str(n_classes), 'train_data', img_name)
#     label_path = os.path.join(data_path, str(n_classes),'train_label', img_name)
#     img = cv2.imread(img_path)
#     label = cv2.imread(label_path)
#     img, label = data_arguementaion(img, label)
#     i, j = random_params(img, img_size)
#     img = img[i:i+img_size, j:j+img_size]
#     label = label[i:i+img_size, j:j+img_size,:1]
#     if img.shape[0]!=img_size or img.shape[1]!=img_size:
#         img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
#         label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)
#         label = label[:,:,np.newaxis]
#         print label.shape
#     img = np.transpose(img, [2, 0, 1])
#     img = Variable(torch.from_numpy(img)).float().cuda()
#     img = torch.unsqueeze(img, 0)
#     label, tmp_gt = get_label(label)
#
#     return img, label, tmp_gt

def convert_vgg( vgg16 ):
    net = vgg()
    vgg_items = net.state_dict().items()
    vgg16_items = vgg16.items()

    pretrain_model = {}
    j = 0
    for k, v in net.state_dict().iteritems():
        v = vgg16_items[j][1]
        k = vgg_items[j][0]
        pretrain_model[k] = v
        j += 1
    return pretrain_model

def random_params(img, output_size):
    h,w, _ = img.shape
    if h<=output_size or w<=output_size:
        i=0
        j=0
    else:
        i = np.random.choice(h - output_size)
        j = np.random.choice(w - output_size)
    return i, j

class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=35),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv2 = nn.Sequential(
            # conv2
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv3 = nn.Sequential(
            # conv3
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv4 = nn.Sequential(
            # conv4
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv5 = nn.Sequential(
            # conv5
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        return conv5


# def get_multi_data(data_path, target_path, position, img_size=256, n_classes=4):
#     left, right, top = position
#     def get_label(label):
#         tmp_gt = label.copy()
#         label = np.transpose(label, [2, 0, 1])
#         label = Variable(torch.from_numpy(label)).long().cuda()
#         return label,tmp_gt
#
#     # get the label
#     label_path = os.path.join(data_path, str(n_classes), 'train_label', target_path)
#     label = cv2.imread(label_path)
#     label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]
#     label, tmp_gt = get_label(label)
#
#     # get the multi_model concat img
#     left_right = {'L': '1', 'R': '0'}
#     target_infor = target_path.split('_')
#     img_id = target_infor[2]
#     copy_path = target_infor[0]+'_'+left_right[target_infor[1]]+'_'+target_infor[-1].split('.')[0]
#     # print copy_path
#     start_idx = 9+int(img_id)*5
#     end_idx = start_idx + 2
#     end_idx_1 = end_idx+1
#     target_img = []
#     for i in range(start_idx,end_idx):
#         if i-start_idx:
#             break
#         img_path = os.path.join(data_path, 'copy_data',copy_path,str(i)+'.jpg')
#         # print img_path
#         tmp_img = cv2.imread(img_path)
#         # print tmp_img.shape
#         tmp_img = tmp_img[top:,left:right]
#         tmp_img = cv2.resize(tmp_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
#         target_img.append(tmp_img)
#     target_img = np.concatenate(target_img,2)
#     target_img = np.transpose(target_img, [2, 0, 1])
#     target_img = Variable(torch.from_numpy(target_img)).float().cuda()
#     target_img = torch.unsqueeze(target_img, 0)
#     return target_img,label,tmp_gt


def get_multi_scale_data(data_path, img_name, img_size=256, n_classes=4):

    def get_data(img, label, new_size):
        img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_AREA)
        img = np.transpose(img, [2, 0, 1])
        img = Variable(torch.from_numpy(img)).float().cuda()
        img = torch.unsqueeze(img, 0)

        label = cv2.resize(label, (new_size, new_size), interpolation=cv2.INTER_AREA)[:, :, :1]
        label = np.transpose(label, [2, 0, 1])
        label = Variable(torch.from_numpy(label)).long().cuda()
        return img, label

    img_path = os.path.join(data_path, str(n_classes), 'train_data', img_name)
    label_path = os.path.join(data_path, str(n_classes), 'train_label', img_name)

    img = cv2.imread(img_path)
    label = cv2.imread(label_path)
    img, label = data_arguementaion(img, label)

    ori_img, ori_label = get_data(img, label, img_size)
    img_75, label_75 = get_data(img, label, int(img_size*0.75))
    img_50, label_50 = get_data(img, label, int(img_size*0.5))
    tmp_gt = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]

    return [ori_img, img_75, img_50], [ori_label, label_75, label_50], tmp_gt




def decode_pixel_label(acc_label):
    label_img = np.zeros([acc_label.shape[0], acc_label.shape[1], 3], dtype=np.uint8)
    for i in range(acc_label.shape[0]):
        for j in range(acc_label.shape[1]):
            if (acc_label[i,j]==1).all():
                label_img[i,j]=[255,255,0]
            elif (acc_label[i,j]==2).all():
                label_img[i,j]=[255,0,0]
            elif (acc_label[i,j]==3).all():
                label_img[i,j]=[0,0,255]

    return label_img

def img_addWeighted(ori_img,pred_img,pred_infor):
    left, right, top = pred_infor['position']
    w,h = pred_infor['size'][:2]
    ROI_img = ori_img[top:, left:right, :]
    pred_img = cv2.resize(pred_img, (h,w),interpolation=cv2.INTER_AREA)

    ROI_img = cv2.addWeighted(ROI_img,0.6,pred_img,0.4,0)
    ori_img[top:, left:right, :] = ROI_img
    return ori_img

def find_boundry(tmp_array):
    max = 0
    min = None
    for i in xrange(1,int(len(tmp_array))/2):
        left = tmp_array[:i]
        right = tmp_array[i:]
        diff = np.sum(right)-np.sum(left)
        if diff >=max:
            max = diff
            left_index = i
        left = tmp_array[:-i]
        right = tmp_array[-i:]
        diff = np.sum(left)-np.sum(right)
        if min is None or min<=diff:
            min = diff
            right_index = len(tmp_array)-i
    return left_index, right_index


def get_top_donw_boundry(ROI_img):
    index_array = np.zeros([ROI_img.shape[1], 2])
    for i in xrange(0,ROI_img.shape[1],30):
        tmp_array = ROI_img[:, i]
        if tmp_array.sum() < 20:
            continue
        left_index, right_index = find_boundry(tmp_array)
        index_array[i] = [left_index, right_index]
    # x = np.linspace(0, ROI_img.shape[0], ROI_img.shape[0])
    x = np.arange(ROI_img.shape[1])
    mask = index_array.sum(axis=1) != 0
    index_array = index_array[mask]
    x = x[mask]
    y1 = index_array[:, 0]
    y2 = index_array[:, 1]

    z1 = np.polyfit(x, y1, 2)
    p1 = np.poly1d(z1)
    plt_x = np.linspace(0, ROI_img.shape[0] - 1, ROI_img.shape[0])
    plt_y_1 = np.polyval(p1, plt_x)

    z2 = np.polyfit(x, y2, 2)
    p2 = np.poly1d(z2)
    plt_y_2 = np.polyval(p2, plt_x)
    return plt_x, plt_y_1, plt_y_2

def crop_boundry(ROI_img, pred_img):
    # plt_x, plt_y_1, plt_y_2 = get_top_donw_boundry(pred_img)
    y_sum = np.sum(pred_img, axis=1)
    mask = y_sum != 0
    y_ = np.arange(pred_img.shape[0])
    y_ = y_[mask]
    y_max = y_[-1]
    y_min = y_[0]
    flag = pred_img.max()==1
    for i in range(y_min,y_max):
        for j in range(pred_img.shape[1]):
            if flag and pred_img[i,j]==1:
                ROI_img[i,j] = [255,255,255]
                ROI_img[i-1, j] = [255, 255, 255]
                ROI_img[i+1, j] = [255, 255, 255]
            elif pred_img[i,j]==1:
                ROI_img[i,j] = [255,255,0]
            elif pred_img[i,j]==2:
                ROI_img[i,j] = [255,0,255]
            elif pred_img[i,j] == 3:
                ROI_img[i,j] = [0,255,255]

    # plt_x, plt_y_1, plt_y_2 = get_top_donw_boundry(ROI_img)
    # new_index_1 = np.stack([plt_x, plt_y_1], 1).astype(np.int32)
    # new_index_2 = np.stack([plt_x, plt_y_2], 1).astype(np.int32)
    # cv2.polylines(ROI_img, [new_index_1], False, [0,0,255], 4)
    # cv2.polylines(ROI_img, [new_index_2], False, [0, 0, 255], 4)
    return ROI_img

def calculate_Accuracy(confusion):
    confusion=np.asarray(confusion)
    pos = np.sum(confusion, 1).astype(np.float32) # 1 for row
    res = np.sum(confusion, 0).astype(np.float32) # 0 for coloum
    tp = np.diag(confusion).astype(np.float32)
    IU = tp / (pos + res - tp)
    meanIU = np.mean(IU)
    pos[pos == 0] = 1
    res[res==0]=1
    pixelAccuracy = np.sum(tp) / np.sum(confusion)
    meanAccuracy = np.mean(tp / pos)
    classAccuracy = np.mean(tp / res)
    return  meanIU,pixelAccuracy,meanAccuracy,classAccuracy,IU


def dice_loss(m1, m2, is_average=True):
    num = m1.size(0)
    m1  = m1.view(num,-1)
    m2  = m2.view(num,-1)
    intersection = (m1 * m2)
    scores = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    if is_average:
        score = scores.sum()/num
        return score
    else:
        return scores

# def my_ployfit(x,y,num,start,end,ratio=2):
#     z1 = np.polyfit(x, y, ratio)
#     p1 = np.poly1d(z1)
#     plt_x = np.linspace(start, end, num)
#     plt_y = np.polyval(p1, plt_x)
#     return plt_x,plt_y

def my_ployfit(x,y,start,end,num = None, ratio=2):
    z1 = np.polyfit(x, y, ratio)
    p1 = np.poly1d(z1)
    if num is None:
        num = int(end-start+1)
    plt_x = np.linspace(start, end, num).astype(np.int32)
    plt_y = np.polyval(p1, plt_x).astype(np.int32)

    return plt_x,plt_y

def process_csv(data):
    for key in data.keys():
        data[key] = data[key].astype(np.float32)
    # data = data.sort_values(by='0')
    r_sum = data.iloc[:, 1:].apply(lambda x: x.sum(), axis=1).values.astype(np.bool)
    data = data.iloc[r_sum, :]
    return data

def process_x_y(csv_data,idx,start,end,flag=False):
    # x = csv_data['0'].values * img.shape[1] / 16.0
    # y = csv_data[str(idx + 1)].values * img.shape[0] / 14.0
    x = csv_data['0'].values * 2130 / 16.0
    y = csv_data[str(idx + 1)].values * 1864 / 14.0
    mask = map(lambda i: not np.isnan(i), y)
    y = y[mask]
    x = x[mask]
    if flag:
        start = x[5]
        end = x[-5:-4]
    x, y = my_ployfit(x, y, num=2130 - 1, start=start, end=end)
    x = np.floor(x)
    y = np.floor(y)
    x,y = new_index(x,y)
    return x,y

def get_truth(ROI_img, img_name, ally, BeginY):
    ori_dir = '/home/intern1/guanghuixu/resnet/data/dataset/eyes/'
    left_right = {'L': '1', 'R': '0'}
    name_infor = img_name.split('_')
    tmp_name = name_infor[:2]
    img_id = name_infor[2]
    if len(name_infor)==4:
        img_id = int(img_id)
        time_data = name_infor[3].split('.')[0]
        tmp_name = tmp_name[0] + '_' + left_right[tmp_name[1]] + '_' + time_data
    else:
        img_id = int(img_id.split('.')[0])
        tmp_name = tmp_name[0] + '_' + left_right[tmp_name[1]]
    csv_dir = os.listdir(os.path.join(ori_dir, tmp_name))
    csv_path = [x for x in csv_dir if x.endswith('.csv')]
    csv_path = os.path.join(ori_dir, tmp_name, csv_path[0])
    csv_data = pd.read_csv(csv_path)

    Lens_front = csv_data[:800]
    Lens_back = csv_data[4015:4815]
    Lens1 = process_csv(Lens_front)
    Lens2 = process_csv(Lens_back)
    Lens2 = Lens2.sort_index(ascending=False)
    Lens = pd.concat([Lens1, Lens2])

    Cortex_front = csv_data[803:1603]
    Cortex_back = csv_data[3212:4012]
    Cortex1 = process_csv(Cortex_front)
    Cortex2 = process_csv(Cortex_back)
    Cortex2 = Cortex2.sort_index(ascending=False)
    Cortex = pd.concat([Cortex1, Cortex2])

    Nucleus_front = csv_data[1606:2406]
    Nucleus_back = csv_data[2409:3209]
    Nucleus1 = process_csv(Nucleus_front)
    Nucleus2 = process_csv(Nucleus_back)
    Nucleus2 = Nucleus2.sort_index(ascending=False)

    for color_id, csv_data in enumerate([[Nucleus1, Nucleus2], [Lens1, Lens2], [Cortex1, Cortex2]]):
        front, back = csv_data
        front_x, front_y = process_x_y(front, img_id, start=ally[0], end=ally[1], flag=True)
        back_x, back_y = process_x_y(back, img_id, start=ally[1], end=ally[0], flag=True)
        # x = np.stack([front_x, back_x]).reshape([-1])
        # y = np.stack([front_y, back_y]).reshape([-1])
        # x = x - ally[0]
        # y = y - BeginY
        front_x = front_x.reshape([-1])
        front_y = front_y.reshape([-1])
        new_index = zip(front_x, front_y)
        new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
        cv2.polylines(ROI_img, [np.int32(new_index)], False, (255, 0, 0), 4)

        back_x = back_x.reshape([-1])
        back_y = back_y.reshape([-1])
        new_index = zip(back_x, back_y)
        new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
        cv2.polylines(ROI_img, [np.int32(new_index)], False, (255, 0, 0), 4)

    return ROI_img

def get_six_truth(tmp_img, img_name, ally):
    gt_list = []
    # ori_dir = './data/dataset/eyes/'
    # left_right = {'L': '1', 'R': '0'}
    # tmp_name = img_name.split('_')[:2]
    # img_id = img_name.split('_')[-1]
    # img_id = int(img_id.split('.')[0])
    # tmp_name = tmp_name[0] + '_' + left_right[tmp_name[1]]
    ori_dir = '/home/intern1/guanghuixu/resnet/data/dataset/eyes/'
    left_right = {'L': '1', 'R': '0'}
    name_infor = img_name.split('_')
    tmp_name = name_infor[:2]
    img_id = name_infor[2]
    if len(name_infor) == 4:
        img_id = int(img_id)
        time_data = name_infor[3].split('.')[0]
        tmp_name = tmp_name[0] + '_' + left_right[tmp_name[1]] + '_' + time_data
    else:
        img_id = int(img_id.split('.')[0])
        tmp_name = tmp_name[0] + '_' + left_right[tmp_name[1]]
    csv_dir = os.listdir(os.path.join(ori_dir, tmp_name))
    csv_path = [x for x in csv_dir if x.endswith('.csv')]
    csv_path = os.path.join(ori_dir, tmp_name, csv_path[0])
    csv_data = pd.read_csv(csv_path)

    Lens_front = csv_data[:800]
    Lens_back = csv_data[4015:4815]
    Lens1 = process_csv(Lens_front)
    Lens2 = process_csv(Lens_back)
    Lens2 = Lens2.sort_index(ascending=False)
    Lens = pd.concat([Lens1, Lens2])

    Cortex_front = csv_data[803:1603]
    Cortex_back = csv_data[3212:4012]
    Cortex1 = process_csv(Cortex_front)
    Cortex2 = process_csv(Cortex_back)
    Cortex2 = Cortex2.sort_index(ascending=False)
    Cortex = pd.concat([Cortex1, Cortex2])

    Nucleus_front = csv_data[1606:2406]
    Nucleus_back = csv_data[2409:3209]
    Nucleus1 = process_csv(Nucleus_front)
    Nucleus2 = process_csv(Nucleus_back)
    Nucleus2 = Nucleus2.sort_index(ascending=False)

    for color_id, csv_data in enumerate([[Nucleus1, Nucleus2], [Cortex1, Cortex2], [Lens1, Lens2]]):
        front, back = csv_data
        front_x, front_y = process_x_y(front, img_id, start=ally[0], end=ally[1], flag=True)
        back_x, back_y = process_x_y(back, img_id, start=ally[1], end=ally[0], flag=True)
        gt_list.append([front_x, front_y, back_x, back_y])
        # x = np.stack([front_x, back_x]).reshape([-1])
        # y = np.stack([front_y, back_y]).reshape([-1])
        front_x = front_x.reshape([-1])
        front_y = front_y.reshape([-1])
        new_index = zip(front_x, front_y)
        new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
        cv2.polylines(tmp_img, [np.int32(new_index)], False, (255, 0, 0))

        back_x = back_x.reshape([-1])
        back_y = back_y.reshape([-1])
        new_index = zip(back_x, back_y)
        new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
        cv2.polylines(tmp_img, [np.int32(new_index)], False, (255, 0, 0))
    # cv2.imwrite('./tmp/%s'%img_name, tmp_img)
    # with open('./tmp/%s.pkl'%img_name.split('.')[0],'w+') as f:
    #     pkl.dump(tmp_img,f)
    # #
    return gt_list


def new_index(x, y):
    index = np.argsort(x)
    x = x[index]
    y = y[index]
    x, index = np.unique(x, return_index=True)
    y = y[index]
    return x, y

def compute_loss(front_x, front_y, front_x_pred, front_y_pred):
    front_x = np.floor(front_x)
    front_y = np.floor(front_y)

    front_x, front_y = new_index(front_x, front_y)
    front_x_pred, front_y_pred = new_index(front_x_pred, front_y_pred)
    min_x = front_x[0]
    max_x = front_x[-1]
    if min_x < front_x_pred[0]:
        min_x = front_x_pred[0]
    if max_x > front_x_pred[-1]:
        max_x = front_x_pred[-1]

    left = np.where(front_x == min_x)[0][0]
    right = np.where(front_x == max_x)[0][0] + 1
    front_x = front_x[left:right]
    front_y = front_y[left:right]

    left = np.where(front_x_pred == min_x)[0][0]
    right = np.where(front_x_pred == max_x)[0][0] + 1
    front_x_pred = front_x_pred[left:right]
    front_y_pred = front_y_pred[left:right]

    ## my_ployfit
    front_x_pred, front_y_pred = my_ployfit(front_x_pred, front_y_pred, max_x - min_x + 1, min_x, max_x)

    loss = np.mean(np.abs(front_y - front_y_pred))
    return loss, front_x, front_y, front_x_pred, front_y_pred

def compute_MSE_pixel(FullImage, img_name, boundry_x_y, ally):
    tmp_img = np.zeros_like(FullImage)
    # front_x, front_y, back_x, back_y = get_truth_annotation(img_name, ally)
    front_x, front_y, back_x, back_y = get_five_annotation(tmp_img, img_name, ally)
    x = boundry_x_y[1]
    y = boundry_x_y[0]
    mean_value = (y.max() - y.min()) / 2 + y.min()
    mask = y < mean_value
    front_y_pred = y[mask]
    front_x_pred = x[mask]
    mask = y >= mean_value
    back_y_pred = y[mask]
    back_x_pred = x[mask]

    front_loss, front_x, front_y, front_x_pred, front_y_pred = compute_loss(front_x, front_y, front_x_pred,
                                                                            front_y_pred)
    back_loss, back_x, back_y, back_x_pred, back_y_pred = compute_loss(back_x, back_y, back_x_pred, back_y_pred)

    new_index = zip(front_x_pred, front_y_pred)
    new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
    cv2.polylines(FullImage, [np.int32(new_index)], False, (255, 0, 0), 4)
    new_index = zip(front_x, front_y)
    new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
    cv2.polylines(FullImage, [np.int32(new_index)], False, (255, 0, 0), 4)
    new_index = zip(back_x_pred, back_y_pred)
    new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
    cv2.polylines(FullImage, [np.int32(new_index)], False, (255, 0, 0), 4)
    new_index = zip(back_x, back_y)
    new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
    cv2.polylines(FullImage, [np.int32(new_index)], False, (255, 0, 0), 4)
    cv2.imwrite('./MSE_pixel/%s'%img_name, FullImage)
    return front_loss, back_loss

def get_boundry_box(boundry):
    w = boundry.shape[1]
    center_x = w / 2
    top_idx = None
    down_idx = None
    new_y = boundry[:, center_x]
    center_y = new_y.shape[0] / 2
    for i in range(1, center_y):
        if new_y[center_y - i]:
            top_idx = center_y - i
            break
    for i in range(1, center_y):
        if new_y[center_y + i]:
            down_idx = center_y + i
            break

    return top_idx, down_idx, center_x

def get_new_boundry(boundry):
    h,w = boundry.shape
    boundry = boundry*1
    new_boundry = np.zeros_like(boundry)
    for i in range(w):
        new_y = boundry[:,i]
        if new_y.sum()<3:
            continue
        tmp_y = np.zeros_like(new_y)
        mask= np.where(new_y==1)[0]
        a = mask[:-1]
        b = mask[1:]
        c = b-a
        new_y = mask[np.where(c!=1)[0]]
        if len(new_y)==5:
            new_y = np.insert(new_y,len(new_y),mask[-1])
        tmp_y[new_y]=1
        new_boundry[:,i] = tmp_y
    return new_boundry

# def findUpAndDown(boundaryMap):
#     whereOne = np.where(boundaryMap == 1)
#     X = whereOne[0]
#     Y = whereOne[1]
#     SortedY = np.sort(Y)
#     SortedY = np.unique(SortedY)
#
#     UpX = []
#     UpY = []
#     DownX = []
#     DownY = []
#
#     for i in range(len(SortedY)):
#         Ytemp = SortedY[i]
#         indexs = np.where(Y==Ytemp)
#         correX = X[indexs]
#         minx = np.min(correX)
#         maxx = np.max(correX)
#
#         UpX.append(minx)
#         UpY.append(Ytemp)
#
#         DownX.append(maxx)
#         DownY.append(Ytemp)
#
#     return UpX,UpY,DownX,DownY

def findUpAndDown(boundaryMap):
    whereOne = np.where(boundaryMap == 1)
    X = whereOne[0]
    Y = whereOne[1]
    SortedY = np.sort(Y)
    # print(Y.shape)

    # # using unique contain a problem
    # SortedY = np.unique(SortedY)

    UpX = []
    UpY = []
    DownX = []
    DownY = []

    # #There is a  problem with  guanghui' code when the boundary is a vertical line, min and max will always select zhe same values
    # for i in range(len(SortedY)):
    #     Ytemp = SortedY[i]
    #     indexs = np.where(Y==Ytemp)
    #     correX = X[indexs]
    #     minx = np.min(correX)
    #     maxx = np.max(correX)
    #
    #     UpX.append(minx)
    #     UpY.append(Ytemp)
    #
    #     DownX.append(maxx)
    #     DownY.append(Ytemp)

    for i in range(len(SortedY)):
        Ytemp = SortedY[i]
        indexs = np.where(Y == Ytemp)
        correX = X[indexs]
        minx = np.min(correX)
        maxx = np.max(correX)
        mid=(minx+maxx)//2
        for j in correX:
            if j <mid:
                UpX.append(j)
                UpY.append(Ytemp)
            else:
                DownX.append(j)
                DownY.append(Ytemp)

    # for i in range(len(SortedY)):
    #     Ytemp = SortedY[i]
    #     indexs = np.where(Y==Ytemp)
    #
    #     if i+1 > len(indexs):
    #         continue
    #
    #     tempmax = -999999
    #     tempmaxindex = 0
    #     tempmin = 999999
    #     tempminindex = 0
    #
    #     # for indexs is tumple(array), so we use indexs[0] instead of indexs
    #     for j in indexs[0]:
    #         if X[j] != -1:
    #             if X[j] > tempmax:
    #                 tempmax = X[j]
    #                 tempmaxindex=j
    #             if X[j] < tempmin:
    #                 tempmin = X[j]
    #                 tempminindex=j
    #
    #     if tempmax != -999999 and tempmin != 999999:
    #         minx=tempmin
    #         maxx=tempmax
    #         X[tempmaxindex]=-1
    #         X[tempminindex]=-1
    #     else:
    #         print('error at findUpAndDown')
    #
    #     UpX.append(minx)
    #     UpY.append(Ytemp)
    #
    #     DownX.append(maxx)
    #     DownY.append(Ytemp)

    return UpX,UpY,DownX,DownY

def compute_abs_loss(X,Y,X_pred,Y_pred):
    newX = X.copy()
    newY = Y.copy()
    for i in range(len(X_pred)):
        xindex = X_pred[i]
        xindex_ = np.where(newX == xindex)
        newY[xindex_] = Y_pred[i]
    loss = np.abs(newY-Y)
    loss = loss[loss!=0]
    loss = np.mean(loss)
    return loss

def MSE_pixel_loss(tmp_img,boundarys,img_name, ally):
    flag=6 # how many line will be compute
    # judge only NucleusShape
    if boundarys.max()==1:
        boundarys[boundarys == 1]=3
        flag = 0
    gt_list = get_six_truth(tmp_img, img_name, ally)
    Nucleus = gt_list[0]
    Cortex = gt_list[1]
    Lens = gt_list[2]

    six_loss = []
    # get nucleus
    NucleusShape = np.zeros(boundarys.shape)
    NucleusShape[boundarys == 3] = 1
    NucleusBoundary = segmentation.find_boundaries(NucleusShape)
    NucleusBoundary = NucleusBoundary * 1
    UpX, UpY, DownX, DownY = findUpAndDown(NucleusBoundary)
    if len(UpX):
        UpX_gt, UpY_gt, DownX_gt, DownY_gt = Nucleus
        loss = compute_abs_loss(UpX_gt, UpY_gt, UpY, UpX)
        six_loss.append(loss)
        loss = compute_abs_loss(DownX_gt, DownY_gt, DownY, DownX)
        six_loss.append(loss)

    LensShape = np.zeros(boundarys.shape)
    LensShape[boundarys == 2] = 1
    LensShape[boundarys == 3] = 1
    LensBoundary = segmentation.find_boundaries(LensShape)
    UpX, UpY, DownX, DownY = findUpAndDown(LensBoundary)
    if len(UpX) and flag:
        UpX_gt, UpY_gt, DownX_gt, DownY_gt = Cortex
        loss = compute_abs_loss(UpX_gt, UpY_gt, UpY, UpX)
        six_loss.append(loss)
        loss = compute_abs_loss(DownX_gt, DownY_gt, DownY, DownX)
        six_loss.append(loss)

    LensShape2 = np.zeros(boundarys.shape)
    LensShape2[boundarys == 1] = 1
    LensShape2[boundarys == 2] = 1
    LensShape2[boundarys == 3] = 1
    LensBoundary2 = segmentation.find_boundaries(LensShape2)
    UpX, UpY, DownX, DownY = findUpAndDown(LensBoundary2)
    if len(UpX) and flag:
        UpX_gt, UpY_gt, DownX_gt, DownY_gt = Lens
        loss = compute_abs_loss(UpX_gt, UpY_gt, UpY, UpX)
        six_loss.append(loss)
        loss = compute_abs_loss(DownX_gt, DownY_gt, DownY, DownX)
        six_loss.append(loss)
    return six_loss





def find_3_region(boundarys,ployfit=False,ratio=3):
    NucleusShape = np.zeros(boundarys.shape)
    NucleusShape[boundarys == 3] = 1
    NucleusBoundary = segmentation.find_boundaries(NucleusShape, mode='inner')

    LensShape = np.zeros(boundarys.shape)
    LensShape[boundarys == 2] = 1
    LensShape[boundarys == 3] = 1
    LensBoundary = segmentation.find_boundaries(LensShape, mode='inner')

    LensShape2 = np.zeros(boundarys.shape)
    LensShape2[boundarys == 1] = 1
    LensShape2[boundarys == 2] = 1
    LensShape2[boundarys == 3] = 1
    LensBoundary2 = segmentation.find_boundaries(LensShape2, mode='inner')

    NucleusBoundary_up = np.zeros_like(NucleusBoundary).astype(np.uint8)
    NucleusBoundary_down = np.zeros_like(NucleusBoundary).astype(np.uint8)
    LensBoundary_up = np.zeros_like(NucleusBoundary).astype(np.uint8)
    LensBoundary_down = np.zeros_like(NucleusBoundary).astype(np.uint8)
    LensBoundary2_up = np.zeros_like(NucleusBoundary).astype(np.uint8)
    LensBoundary2_down = np.zeros_like(NucleusBoundary).astype(np.uint8)

    for i,region in enumerate([NucleusBoundary, LensBoundary, LensBoundary2]):
        out_up = np.zeros_like(NucleusBoundary).astype(np.uint8)
        out_down = np.zeros_like(NucleusBoundary).astype(np.uint8)

        # y is axis 0, x is axis 1,the original output is UpX, UpY, DownX, DownY = findUpAndDown(LensBoundary2)
        UpY, UpX, DownY, DownX = findUpAndDown(region)

        if ployfit:
            UpX, UpY = my_ployfit(UpX, UpY, start=UpX[0], end=UpX[-1], ratio=ratio)
            DownX, DownY = my_ployfit(DownX, DownY, start=DownX[0], end=DownX[-1], ratio=ratio)

        for idx,(tmp_x,tmp_y) in enumerate(zip(UpX,UpY)):
            # in case tmp_y<0
            if tmp_y<=0:
                tmp_y=0
            if tmp_y>=out_down.shape[0]:
                tmp_y=out_down.shape[0]-1
            out_up[tmp_y,tmp_x]=1
        for idx,(tmp_x,tmp_y) in enumerate(zip(DownX,DownY)):
            # in case tmp_y is out of boundary
            if tmp_y>=out_down.shape[0]:
                tmp_y=out_down.shape[0]-1
            if tmp_y<=0:
                tmp_y=0
            out_down[tmp_y,tmp_x]=1

        if i==0:
            NucleusBoundary_up=out_up
            NucleusBoundary_down=out_down
        elif i==1:
            LensBoundary_up=out_up
            LensBoundary_down=out_down
        else:
            LensBoundary2_up=out_up
            LensBoundary2_down=out_down


    NucleusBoundary_up=NucleusBoundary_up.astype(np.bool)
    NucleusBoundary_down = NucleusBoundary_down.astype(np.bool)
    LensBoundary_up = LensBoundary_up.astype(np.bool)
    LensBoundary_down = LensBoundary_down.astype(np.bool)
    LensBoundary2_up = LensBoundary2_up.astype(np.bool)
    LensBoundary2_down = LensBoundary2_down.astype(np.bool)

    if ployfit:
        NucleusBoundary=NucleusBoundary_up+NucleusBoundary_down
        LensBoundary=LensBoundary_up+LensBoundary_down
        LensBoundary2=LensBoundary2_up+LensBoundary2_down

    # NucleusBoundary = NucleusBoundary_up + NucleusBoundary_down
    # LensBoundary = LensBoundary_up + LensBoundary_down
    # LensBoundary2 = LensBoundary2_up + LensBoundary2_down

    return NucleusBoundary, LensBoundary, LensBoundary2, NucleusBoundary_up, NucleusBoundary_down, LensBoundary_up, LensBoundary_down, LensBoundary2_up, LensBoundary2_down


def compute_not_csv_loss(output, path, data_path,use_crop=False,ployfit=False):
    img_name=path
    path = path[:-3]+'npy'
    # path = os.path.join('/home/intern1/guanghuixu/resnet/data/new_data_1/split_distance',str(size),path)
    path = os.path.join(data_path+'/distance',path)
    # path = os.path.join(data_path + '/distance_resize256', path)
    distance_map = np.load(path)
    # distance_map = np.load(path).min(axis=0)

    if use_crop:
        with open(os.path.join(data_path, 'croped_up_and_down_dict.pkl')) as f:
            croped_up_and_down_dict = pkl.load(f)
        [h1,h2]=croped_up_and_down_dict[img_name]
        distance_map=distance_map[:,h1:h2,:]

    _, height, width = distance_map.shape
    # print output.shape,output.max()
    output = cv2.resize(output.astype(np.uint8), (width, height))
    # print output.shape

    NucleusBoundary, LensBoundary, LensBoundary2, NucleusBoundary_up, NucleusBoundary_down, LensBoundary_up, LensBoundary_down, LensBoundary2_up, LensBoundary2_down = find_3_region(output,ployfit)



    # print distance_map.shape,NucleusBoundary.shape
    loss_1 = distance_map[2][NucleusBoundary]
    loss_2 = distance_map[1][LensBoundary]
    loss_3 = distance_map[0][LensBoundary2]
    loss_NucleusBoundary_up=distance_map[2][NucleusBoundary_up]
    loss_NucleusBoundary_down = distance_map[2][NucleusBoundary_down]
    loss_LensBoundary_up=distance_map[1][LensBoundary_up]
    loss_LensBoundary_down = distance_map[1][LensBoundary_down]
    loss_LensBoundary2_up=distance_map[0][LensBoundary2_up]
    loss_LensBoundary2_down = distance_map[0][LensBoundary2_down]

    return loss_1,loss_2,loss_3,loss_NucleusBoundary_up,loss_NucleusBoundary_down,loss_LensBoundary_up,loss_LensBoundary_down,loss_LensBoundary2_up,loss_LensBoundary2_down

def compute_not_csv_loss_use_resize(output, path, data_path,img_size):
    path = path[:-3]+'npy'
    # path = os.path.join('/home/intern1/guanghuixu/resnet/data/new_data_1/split_distance',str(size),path)
    path = os.path.join(data_path+'/distance',path)
    distance_map = np.load(path)
    # distance_map = np.load(path).min(axis=0)
    _, height, width = distance_map.shape
    # change the distance map to size(img_size,img_size),
    distance_map=np.transpose(distance_map,[1,2,0])
    distance_map=cv2.resize(distance_map.astype(np.uint8),(img_size,img_size),interpolation=cv2.INTER_AREA)
    distance_map = np.transpose(distance_map, [2,0,1])
    # distance_map[0]=cv2.resize(distance_map[0].astype(np.uint8),(img_size,img_size),interpolation=cv2.INTER_AREA)
    # distance_map[1] = cv2.resize(distance_map[1].astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_AREA)
    # distance_map[2] = cv2.resize(distance_map[2].astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_AREA)


    NucleusBoundary, LensBoundary, LensBoundary2 = find_3_region(output)
    # print distance_map.shape,NucleusBoundary.shape
    loss_1 = distance_map[2][NucleusBoundary]
    loss_2 = distance_map[1][LensBoundary]
    loss_3 = distance_map[0][LensBoundary2]
    return loss_1,loss_2,loss_3

def get_coco_data(detection, img_idx, gpu_idx=None):

    img_list = []
    target_list = []
    tmp_label_list = []
    for i in img_idx:
        img, target, tmp_img, tmp_label = detection[i]
        if len(tmp_label)==0:
            return [], [], [], []

        if gpu_idx:
            img = img.cuda()
            target = target.cuda()
            # img = img.cuda(gpu_idx)
            # target = target.cuda(gpu_idx)

        img_list.append(img)
        target_list.append(target)
        tmp_label_list.append(tmp_label)

    img = torch.cat(img_list, 0)
    target = torch.cat(target_list, 0)
    tmp_label = np.stack(tmp_label_list, 0)

    return img, target, tmp_img, tmp_label

colors = [
        (255, 182, 193), (255, 192, 203), (220, 20, 60), (255, 240, 245), (219, 112, 147), (255, 105, 180),
        (255, 20, 147), (199, 21, 133), (218, 112, 214), (216, 191, 216), (221, 160, 221), (238, 130, 238),
        (255, 0, 255), (139, 0, 139), (128, 0, 128), (186, 85, 211), (148, 0, 211), (153, 50, 204), (75, 0, 130),
        (138, 43, 226), (147, 112, 219), (123, 104, 238), (106, 90, 205), (72, 61, 139), (230, 230, 250),
        (248, 248, 255), (0, 0, 255), (0, 0, 205), (25, 25, 112), (0, 0, 139), (0, 0, 128), (65, 105, 225),
        (100, 149, 237), (176, 196, 222), (119, 136, 153), (112, 128, 144), (30, 144, 255), (240, 248, 255),
        (70, 130, 180), (135, 206, 250), (135, 206, 235), (0, 191, 255), (173, 216, 230), (176, 224, 230),
        (95, 158, 160), (240, 255, 255), (225, 255, 255), (175, 238, 238), (0, 255, 255), (0, 255, 255),
        (0, 206, 209), (47, 79, 79), (0, 139, 139), (0, 128, 128), (72, 209, 204), (32, 178, 170),
        (64, 224, 208), (127, 255, 170), (0, 250, 154), (245, 255, 250), (0, 255, 127), (60, 179, 113),
        (46, 139, 87), (240, 255, 240), (144, 238, 144), (152, 251, 152), (143, 188, 143), (50, 205, 50),
        (0, 255, 0), (34, 139, 34), (0, 128, 0), (0, 100, 0), (127, 255, 0), (124, 252, 0), (173, 255, 47),
        (85, 107, 47), (107, 142, 35), (250, 250, 210), (255, 255, 240), (255, 255, 224), (255, 255, 0),
        (128, 128, 0), (189, 183, 107), (255, 250, 205), (238, 232, 170), (240, 230, 140), (255, 215, 0),
        (255, 248, 220), (218, 165, 32), (255, 250, 240), (253, 245, 230), (245, 222, 179), (255, 228, 181),
        (255, 165, 0), (255, 239, 213), (255, 235, 205), (255, 222, 173), (250, 235, 215), (210, 180, 140),
        (222, 184, 135), (255, 228, 196), (255, 140, 0), (250, 240, 230), (205, 133, 63), (255, 218, 185),
        (244, 164, 96), (210, 105, 30), (139, 69, 19), (255, 245, 238), (160, 82, 45), (255, 160, 122),
        (255, 127, 80), (255, 69, 0), (233, 150, 122), (255, 99, 71), (255, 228, 225), (250, 128, 114),
        (255, 250, 250), (240, 128, 128), (188, 143, 143), (205, 92, 92), (255, 0, 0), (165, 42, 42),
        (178, 34, 34), (139, 0, 0), (128, 0, 0), (255, 255, 255), (245, 245, 245), (220, 220, 220),
        (211, 211, 211), (192, 192, 192), (169, 169, 169), (128, 128, 128), (105, 105, 105), (0, 0, 0)
    ]
def get_cur_color(img):
    h,w = img.shape
    tmp_img = np.zeros([h,w,3], dtype=np.uint8)
    for i in xrange(h):
        for j in xrange(w):
            tmp_img[i,j] = colors[img[i,j]]
    return tmp_img

def get_cur_model(model_name, n_classes, bn=True, pretrain=False):
    if model_name=='UNet128':
        return UNet128(n_classes=n_classes, bn=bn)
    elif model_name=='UNet256':
        return UNet256(n_classes=n_classes, bn=bn)
    elif model_name=='UNet512':
        return UNet512(n_classes=n_classes, bn=bn)
    elif model_name=='UNet1024':
        return UNet1024(n_classes=n_classes, bn=bn)
    elif model_name=='UNet512_SideOutput':
        return UNet512_SideOutput(n_classes=n_classes, bn=bn)
    elif model_name=='UNet1024_SideOutput':
        return UNet1024_SideOutput(n_classes=n_classes, bn=bn)
    elif model_name=='resnet_50':
        return resnet_50(n_classes,pretrain=pretrain)
    elif model_name=='resnet_dense':
        return resnet_dense(n_classes,pretrain=pretrain)
    elif model_name=='UNet128_deconv':
        return UNet128_deconv(n_classes=n_classes, bn=bn)
    elif model_name=='FPN18':
        return FPN18(n_classes=n_classes, pretrained=pretrain)
    elif model_name=='UNet1024_deconv':
        return UNet1024_deconv(n_classes=n_classes, bn=bn)
    elif model_name=='FPN_deconv':
        return FPN_deconv(n_classes=n_classes, bn=bn)
    elif model_name=='My_UNet':
        return My_UNet(n_classes=n_classes, bn=bn)
    elif model_name=='ModelUNetTogether':
        return UNet(n_classes=n_classes, bn=bn)
    elif model_name=='multi_scale':
        return UNet1024(n_classes=n_classes, bn=bn)
    elif model_name=='M_Net':
        return M_Net(n_classes=n_classes, bn=bn)
    elif model_name=='M_Net_deconv':
        return M_Net_deconv(n_classes=n_classes, bn=bn)
    elif model_name=='Small':
        return Small(n_classes=n_classes, bn=bn)
    elif model_name=='VGG':
        return VGG(n_classes=n_classes, bn=bn)
    elif model_name=='BNM':
        return BNM(n_classes=n_classes, bn=bn)
    elif model_name=='Guanghui':
        return Guanghui(n_classes=n_classes, bn=bn)
    elif model_name=='GuanghuiXu':
        return GuanghuiXu(n_classes=n_classes, bn=bn)
    elif model_name=='Multi_Model':
        return Multi_Model(n_classes=n_classes, bn=bn)
    elif model_name=='Patch_Model':
        return UNet1024(n_classes=n_classes, bn=bn)
    elif model_name=='HED':
        model = HED()
        # pretrained_dict = torch.load('./models/UNet1024/vgg16.pth')
        # pretrained_dict = convert_vgg(pretrained_dict)
        # model_dict = model.state_dict()
        # # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # # 3. load the new state dict
        # model.load_state_dict(model_dict)
        # print ('copy the weight sucessfully')
        return model
    elif model_name=='finetune':
        return UNet1024(n_classes=n_classes, bn=bn)
    elif model_name=='BNM_1':
        return BNM_1(n_classes=n_classes, bn=bn)
    elif model_name=='BNM_2':
        return BNM_2(n_classes=n_classes, bn=bn)
    elif model_name=='BNM_3':
        return BNM_3(n_classes=n_classes, bn=bn)
    elif model_name=='level_set':
        return UNet512(n_classes=n_classes, bn=bn)
    elif model_name=='GroupNorm':
        return GroupNorm(n_classes=n_classes, bn=bn)
    elif model_name=='MobileNet':
        return MobileNet(n_classes=n_classes, bn=bn)


def bce2d(input, target):
    n, c, h, w = input.size()
    # assert(max(target) == 1)
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()
    pos_index = (target_t > 0)
    neg_index = (target_t == 0)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0
    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)
    return loss

def criterion(softmax_2d, logits_list, labels_list, n_class):
    loss_list = []
    out_1, out_2, out_3, out_4, out_5, out_6 = logits_list
    label_1, label_2, label_3, label_4, label_5, label_6 = labels_list

    out_1 = torch.log(softmax_2d(out_1))
    out_2 = torch.log(softmax_2d(out_2))
    out_3 = torch.log(softmax_2d(out_3))
    out_4 = torch.log(softmax_2d(out_4))
    out_5 = torch.log(softmax_2d(out_5))
    out_6 = torch.log(softmax_2d(out_6))

    loss_1 = lossfunc(out_1, label_1[0])
    loss_2 = lossfunc(out_2, label_2[0])
    loss_3 = lossfunc(out_3, label_3[0])
    loss_4 = lossfunc(out_4, label_4[0])
    loss_5 = lossfunc(out_5, label_5[0])
    loss_6 = lossfunc(out_6, label_6[0])

    l = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6

    loss_list.append(l)
    ppi_1 = np.argmax(out_1.cpu().data.numpy(), 1).reshape((img_size, img_size))
    ppi_2 = np.argmax(out_2.cpu().data.numpy(), 1).reshape((img_size/2, img_size/2))
    ppi_3 = np.argmax(out_3.cpu().data.numpy(), 1).reshape((img_size/4, img_size/4))
    ppi_4 = np.argmax(out_4.cpu().data.numpy(), 1).reshape((img_size/8, img_size/8))
    ppi_5 = np.argmax(out_5.cpu().data.numpy(), 1).reshape((img_size/16, img_size/16))
    ppi_6 = np.argmax(out_6.cpu().data.numpy(), 1).reshape((img_size/32, img_size/32))

    confusion = np.zeros([n_class, n_class])

    def compute_acc(confusion,tmp_gt,ppi):
        tmp_gt = tmp_gt.reshape([-1])

        tmp_out = ppi.reshape([-1])
        for idx in xrange(len(tmp_gt)):
            confusion[tmp_gt[idx], tmp_out[idx]] += 1
        return confusion

    confusion = compute_acc(confusion,label_1[1],ppi_1)
    confusion = compute_acc(confusion, label_2[1], ppi_2)
    confusion = compute_acc(confusion, label_3[1], ppi_3)
    confusion = compute_acc(confusion, label_4[1], ppi_4)
    confusion = compute_acc(confusion, label_5[1], ppi_5)
    confusion = compute_acc(confusion, label_6[1], ppi_6)

    meanIU, pixelAccuracy, meanAccuracy, classAccuracy = calculate_Accuracy(confusion)

    return l, meanIU, pixelAccuracy, meanAccuracy, classAccuracy

def new_ployfit(x,y,num,start,end,ratio=2):
    z1 = np.polyfit(x, y, ratio)
    p1 = np.poly1d(z1)
    plt_x = np.linspace(start, end, num)
    plt_y = np.polyval(p1, plt_x)
    return z1,plt_x,plt_y

def compute_intersection(z_1, z_2):
    a, b, c = z_1 - z_2
    if a < 1e-8:
        return 0, 0
    deta = b ** 2 - (4 * a * c)
    if deta < 1e-8:
        return -b / (2 * a), -b / (2 * a)
    elif deta > 0:
        return (-b - np.sqrt(deta)) / (2 * a), (-b + np.sqrt(deta)) / (2 * a)
    elif deta < 0:
        return 0, 0

def choice_leves_set_shape(data):
    image = np.argmax(data, 1)[0]
    boundry = segmentation.find_boundaries(image)
    boundry = boundry * 1
    h = boundry.shape[0]
    UpY, UpX, DownY, DownX = findUpAndDown(boundry)
    z_1, tmp_front_x, tmp_front_y = new_ployfit(np.array(UpX), np.array(UpY), num=2130 - 1, start=0, end=h)
    z_2, tmp_back_x, tmp_back_y = new_ployfit(np.array(DownX), np.array(DownY), num=2130 - 1, start=h, end=0)
    ans_1, ans_2 = compute_intersection(z_1, z_2)
    print(ans_1, ans_2)
    if ans_1<0 or ans_2>h:
        return 1
    elif ans_1 >= 0 and ans_2 <= h:
        return 2
    else:
        return 3

def reconstruct_img(output_img, train_img,img_size=1024,random=40):
    h = output_img.shape[0]
    # print output_img.max()
    nurcles = np.zeros_like(output_img)
    nurcles [output_img==3]=1
    boundry = segmentation.find_boundaries(nurcles)
    boundry =boundry*1
    UpY, UpX, DownY, DownX = findUpAndDown(boundry)
    z_1, tmp_front_x, tmp_front_y = new_ployfit(np.array(UpX), np.array(UpY)+random, num=2130 - 1, start = 0, end=h)
    z_2, tmp_back_x, tmp_back_y = new_ployfit(np.array(DownX), np.array(DownY)-random, num=2130 - 1, start = h, end=0)
    ans_1, ans_2 = compute_intersection(z_1, z_2)
    mask = tmp_front_x > ans_1
    tmp_front_x = tmp_front_x[mask]
    tmp_front_y = tmp_front_y[mask]
    mask = tmp_front_x < ans_2
    tmp_front_x = tmp_front_x[mask]
    tmp_front_y = tmp_front_y[mask]

    mask = tmp_back_x > ans_1
    tmp_back_x = tmp_back_x[mask]
    tmp_back_y = tmp_back_y[mask]
    mask = tmp_back_x < ans_2
    tmp_back_x = tmp_back_x[mask]
    tmp_back_y = tmp_back_y[mask]
    x = np.stack([tmp_front_x, tmp_back_x]).reshape([-1])
    y = np.stack([tmp_front_y, tmp_back_y]).reshape([-1])
    tmp_index = zip(x, y)
    tmp_index = np.array(tmp_index, np.int32).reshape([-1, 1, 2])
    train_img = cv2.resize(train_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    train_img = cv2.fillPoly(train_img, [tmp_index], [127,127,127])
    return train_img

def crop_img_label(output_img, img_name, img, img_size=1024, random=50, flag='train'):

    nurcles = np.zeros_like(output_img)
    nurcles[output_img == 3] = 1
    # print nurcles.sum()
    boundry = segmentation.find_boundaries(nurcles)
    boundry = boundry*1
    y,x = np.where(boundry==1)
    # print y
    max_y,min_y = y.max(),y.min()
    label_path = os.path.join('/home/intern1/guanghuixu/resnet/data/dataset', '4', 'train_label', img_name)

    label = cv2.imread(label_path)
    img, label = data_arguementaion(img, label)
    tmp_idx = map(lambda x:max(x,0), [min_y-random])[0]
    img = img[int(tmp_idx):int(max_y+random)]
    # print img.shape, tmp_idx, max_y
    label = label[int(tmp_idx):int(max_y + random)]
    height = tmp_idx+label.shape[0]
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    label = cv2.resize(label, (img_size, img_size), interpolation=cv2.INTER_AREA)[:, :, :1]
    # cv2.imwrite('./logs/test_data/test.png',img)
    # cv2.imwrite('./logs/test_data/test_label.png',label)
    img = np.transpose(img, [2, 0, 1])
    img = Variable(torch.from_numpy(img)).float().cuda()
    img = torch.unsqueeze(img, 0)

    # label, tmp_gt = get_label(label)

    tmp_boundry = segmentation.find_boundaries(label)
    tmp_boundry = tmp_boundry.astype(np.uint8)
    tmp_boundry = np.transpose(tmp_boundry, [2, 0, 1])
    tmp_boundry = tmp_boundry[np.newaxis, :]

    tmp_boundry = torch.from_numpy(tmp_boundry.astype(np.float32))
    tmp_boundry = Variable(tmp_boundry).cuda()

    return img, tmp_boundry, tmp_idx, height

def fit_ellipse(ROI_img, FullImage):
    tmp_img = np.zeros_like(FullImage)
    tmp_img[FullImage == 3] = 1
    # plt.imshow(tmp_img)
    boundry = segmentation.find_boundaries(tmp_img)
    boundry = boundry * 1
    y, x = np.where(boundry == 1)
    new_idx = [zip(x, y)]
    seg = FullImage
    slic, points_all, labels = tl_fit.get_slic_points_labels(seg, slic_size=15, slic_regul=0.3)
    # for lb in np.unique(labels):
    #     plt.plot(points_all[labels == lb, 1], points_all[labels == lb, 0], '.')
    # _ = plt.xlim([0, seg.shape[1]]), plt.ylim([seg.shape[0], 0])
    weights = np.bincount(slic.ravel())
    ellipses, crits = [], []
    TABLE_FB_PROBA = [[0.01, 0.7, 0.95, 0.8],
                      [0.99, 0.3, 0.05, 0.2]]
    COLORS = 'bgrmyck'
    for i, points in enumerate(new_idx):
        # for x, y in points:
        #     plt.scatter(x, y, 3)
        model, _ = tl_fit.ransac_segm(points, tl_fit.EllipseModelSegm, points_all, weights, labels,
                                      TABLE_FB_PROBA, min_samples=0.4, residual_threshold=10, max_trials=150)
        if model is None: continue
        c1, c2, h, w, phi = model.params
        ellipses.append(model.params)
        crit = model.criterion(points_all, weights, labels, TABLE_FB_PROBA)
        crits.append(np.round(crit))
        print ('model params:', (int(c1), int(c2), int(h), int(w), phi))
        print ('-> crit:', model.criterion(points_all, weights, labels, TABLE_FB_PROBA))
        rr, cc = tl_visu.ellipse_perimeter(int(c1), int(c2), int(h), int(w), phi)
        # plt.plot(rr, cc, '.', color=COLORS[i % len(COLORS)])
        front_x = rr.reshape([-1])
        front_y = cc.reshape([-1])
        new_index = zip(front_x, front_y)
        new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
        cv2.polylines(ROI_img, [np.int32(new_index)], False, (255, 0, 0))
    return ROI_img

def ray_feature(seg):
    df = pd.read_csv(os.path.join('logs', 'ray_shapes.csv'), index_col=0)
    list_rays = df.values
    x_axis = np.linspace(0, 360, list_rays.shape[1], endpoint=False)
    list_cdf = tl_rg.transform_rays_model_cdf_histograms(list_rays, nb_bins=25)
    cdist = np.array(list_cdf)
    seg_object = (seg == 1)
    centre = ndimage.measurements.center_of_mass(seg_object)
    ray = tl_fts.compute_ray_features_segm_2d(seg_object, centre, edge='down')
    _, shift = tl_fts.shift_ray_features(ray)
    prior_map = np.zeros(seg_object.shape)
    error_pos = []
    for i in np.arange(prior_map.shape[0], step=5):
        for j in np.arange(prior_map.shape[1], step=5):
            prior_map[i:i + 5, j:j + 5] = tl_rg.compute_shape_prior_table_cdf([i, j], cdist, centre, angle_shift=shift)

def data_arguementaion(image,label):
    gamma = np.random.uniform(0.5,1)
    flip_p = np.random.rand()
    if flip_p > 0.5:
        image = np.fliplr(image)
        label = np.fliplr(label)

    image = exposure.adjust_gamma(image, gamma)

    return image, label

def model_transform(model_name):
    return torch.load(model_name, map_location=lambda storage, loc:storage)           # gpu-->cpu
    # return torch.load(model_name, map_location={'cuda:1':'cuda:0'})                 # gpu1-->cpu0
    # return torch.load(model_name, map_location=lambda storage, loc:storage.cuda(1)) # cpu-->gpu1


def calculate_density(output,image):
    h,w = output.shape
    image1 = image.cpu()
    nucleus_len=cortex_len=lens_len=0
    nucleus_density=cortex_density=lens_density=0
    for i in range(h):
        for j in range(w):
            if output[i][j] ==3:
                nucleus_density += image1.data[0][0][i][j].numpy()
                nucleus_len +=1
            elif output[i][j] == 2:
                cortex_density += image1.data[0][0][i][j].numpy()
                cortex_len += 1
            elif output[i][j] == 1:
                lens_density += image1.data[0][0][i][j].numpy()
                lens_len += 1

    return nucleus_density/nucleus_len, cortex_density/cortex_len, lens_density/lens_len


# def calculate_density_label(data_path, path):
#     img_path = os.path.join(data_path, 'train_data', path)
#     label_path = os.path.join(data_path, 'train_label', path)
#     # print img_path
#
#     img = cv2.imread(img_path)
#     label = cv2.imread(label_path, 0)
#
#     h,w=label.shape
#     nucleus_len=cortex_len=lens_len=0
#     nucleus_density=cortex_density=lens_density=0
#     for i in range(h):
#         for j in range(w):
#             if label[i][j] ==3:
#                 nucleus_density += img[i][j][0]
#                 nucleus_len +=1
#             elif label[i][j] == 2:
#                 cortex_density += img[i][j][0]
#                 cortex_len += 1
#             elif label[i][j] == 1:
#                 lens_density += img[i][j][0]
#                 lens_len += 1
#
#     return nucleus_density/nucleus_len, cortex_density/cortex_len, lens_density/lens_len

def calculate_density_label(data_path, path, fiag_16bit=False):
    img_path = os.path.join(data_path, 'train_data', path)
    label_path = os.path.join(data_path, 'train_label', path)
    # print img_path

    if fiag_16bit:
        img_path = os.path.join(data_path, 'train_data', path[:-4]+'.mat')
        img = scio.loadmat(img_path)['mm']
    else:
        img = cv2.imread(img_path)

    label = cv2.imread(label_path, 0)

    nucleus_len = cortex_len = lens_len = 0
    nucleus_density = cortex_density = lens_density = 0

    img = img[:, :, 0]

    A = np.where(label == 3)
    B = img[A]
    nucleus_len +=len(B)
    nucleus_density +=np.sum(B)

    A = np.where(label == 2)
    B = img[A]
    cortex_len +=len(B)
    cortex_density +=np.sum(B)

    A = np.where(label == 1)
    B = img[A]
    lens_len +=len(B)
    lens_density +=np.sum(B)

    return nucleus_density / nucleus_len, cortex_density / cortex_len, lens_density / lens_len


def calculate_statistical_information(data_path, path, fiag_16bit=False):
    """
    Get density information about nucleus, cortex, lens
    Author: Shihao Zhang
    :param data_path: root path about the data
    :param path:  image name
    :param fiag_16bit: Weather the image data is 16 bits and stored with .mat format
    :return: information about nucleus, cortex, lens with array
    """
    img_path = os.path.join(data_path, 'train_data', path)
    label_path = os.path.join(data_path, 'train_label', path)
    # print img_path

    if fiag_16bit:
        img_path = os.path.join(data_path, 'train_data', path[:-4] + '.mat')
        img = scio.loadmat(img_path)['mm']

    else:
        img = cv2.imread(img_path)

    label = cv2.imread(label_path, 0)
    img = img[:, :, 0]

    # ---------- calculate ----------

    # nucleus
    nucleus_xy = np.where(label == 3)
    nucleus_x = nucleus_xy[0]
    nucleus_y = nucleus_xy[1]

    # get middle coordinate
    middle_up_down = (np.max(nucleus_x)+np.min(nucleus_x))//2
    middle_left_rigth = (np.max(nucleus_y)+np.min(nucleus_y))//2

    # calculate total_values, up_values, down_values
    nucleus_value = img[nucleus_xy]
    tmp = np.where(nucleus_x >= middle_up_down)
    nucleus_up_value = img[nucleus_x[tmp], nucleus_y[tmp]]
    tmp = np.where(nucleus_x < middle_up_down)
    nucleus_down_value = img[nucleus_x[tmp], nucleus_y[tmp]]

    # cortex
    cortex_xy = np.where(label == 2)
    cortex_x = cortex_xy[0]
    cortex_y = cortex_xy[1]
    # values
    cortex_value = img[cortex_xy]
    tmp = np.where(cortex_x >= middle_up_down)
    cortex_up_value = img[cortex_x[tmp], cortex_y[tmp]]
    tmp = np.where(cortex_x < middle_up_down)
    cortex_down_value = img[cortex_x[tmp], cortex_y[tmp]]

    # lens
    lens_xy = np.where(label == 1)
    lens_x = lens_xy[0]
    lens_y = lens_xy[1]
    # values
    lens_value = img[lens_xy]
    tmp = np.where(lens_x >= middle_up_down)
    lens_up_value = img[lens_x[tmp], lens_y[tmp]]
    tmp = np.where(lens_x < middle_up_down)
    lens_down_value = img[lens_x[tmp], lens_y[tmp]]

    return nucleus_value, nucleus_up_value, nucleus_down_value,\
           cortex_value, cortex_up_value, cortex_down_value,\
           lens_value, lens_up_value, lens_down_value



# def get_model(model_name):
#     if model_name=='UNet128':
#         return UNet128
#     elif model_name=='UNet256':
#         return UNet256
#     elif model_name=='UNet512':
#         return UNet512
#     elif model_name=='UNet1024':
#         return UNet1024
#     elif model_name=='UNet256_kernel':
#         return UNet256_kernel
#     elif model_name=='UNet512_kernel':
#         return UNet512_kernel
#     elif model_name=='UNet1024_kernel':
#         return UNet1024_kernel
#     elif model_name=='UNet256_kernel_dgf':
#         return UNet256_kernel_dgf
#     elif model_name=='UGF':
#         return UGF
#     elif model_name=='UNet256_kernel_classification':
#         return UNet256_kernel_classification
#     elif model_name=='resnet_50':
#         return resnet50
#     elif model_name =='UNet256_kernel_label':
#         return UNet256_kernel_label
#     elif model_name =='UNet256_kernel_figure':
#         return UNet256_kernel_figure
#     elif model_name =='M_Net':
#         return M_Net
#     elif model_name =='GM':
#         return GM
#     elif model_name =='G_MM':
#         return G_MM
#     elif model_name == 'G_MM_1':
#         return G_MM_1
#     elif model_name == 'G_MM_2':
#         return G_MM_2
#     elif model_name == 'G_MM_3':
#         return G_MM_3
#     elif model_name == 'GF':
#         return GF
#     elif model_name == 'deeplab_resnet':
#         return deeplab_resnet.Res_Deeplab()
#     elif model_name == 'FCN':
#         A = fcn.FCN8s(n_class=4)
#         return A
#     elif model_name=='PSP':
#         return pspnet.PSPNet(psp_size=512, deep_features_size=256)
#     else:
#         print('no such model')

def get_model(model_name):
    if model_name=='UNet128':
        return UNet128
    elif model_name=='UNet256':
        return UNet256
    elif model_name=='UNet512':
        return UNet512
    elif model_name=='UNet1024':
        return UNet1024
    elif model_name=='UNet256_kernel':
        return UNet256_kernel
    elif model_name=='UNet512_kernel':
        return UNet512_kernel
    elif model_name=='UNet1024_kernel':
        return UNet1024_kernel
    elif model_name=='UNet256_kernel_dgf':
        return UNet256_kernel_dgf
    elif model_name=='UGF':
        return UGF
    elif model_name=='UNet256_kernel_classification':
        return UNet256_kernel_classification
    elif model_name=='resnet_50':
        return resnet50
    elif model_name =='UNet256_kernel_label':
        return UNet256_kernel_label
    elif model_name =='UNet256_kernel_figure':
        return UNet256_kernel_figure
    elif model_name =='M_Net':
        return M_Net
    elif model_name =='GM':
        return GM
    elif model_name =='G_MM':
        return G_MM
    elif model_name == 'G_MM_1':
        return G_MM_1
    elif model_name == 'G_MM_2':
        return G_MM_2
    elif model_name == 'G_MM_3':
        return G_MM_3
    elif model_name == 'GF':
        return GF
    elif model_name == 'deeplab_resnet':
        print('deeplab_resnet')
    elif model_name =='G_MM_1_modified':
        return G_MM_1_modified
    elif model_name =='G_MM_1_modified2':
        return G_MM_1_modified2
    elif model_name =='G_MM_1_modified3':
        return G_MM_1_modified3
    elif model_name =='G_N':
        return G_N
    elif model_name == 'G_MM_1_new':
        return G_MM_1_new
    elif model_name == 'G_HuaZhu':
        return G_HuaZhu
    elif model_name == 'G_HuaZhu_2':
        return G_HuaZhu_2
    elif model_name == 'G_Up':
        return G_Up
    elif model_name =='M_Net_Up':
        return M_Net_Up
    elif model_name =='M_Net_Up_2':
        return M_Net_Up_2
    elif model_name =='M_Net_Up_3':
        return M_Net_Up_3
    elif model_name =='G_MM_compare':
        return G_MM_compare
    elif model_name == 'G_HuaZhu_3':
        return G_HuaZhu_3
    elif model_name == 'G_HuaZhu_4':
        return G_HuaZhu_4
    elif model_name =='M_Net_Up_4':
        return M_Net_Up_4
    elif model_name =='M_Net_Up_5':
        return M_Net_Up_5
    elif model_name =='M_Net_Up_6':
        return M_Net_Up_6
    elif model_name =='M_Net_Up_7':
        return M_Net_Up_7
    elif model_name =='G_MM_New_1':
        return G_MM_New_1
    elif model_name =='G_MM_New_2':
        return G_MM_New_2
    elif model_name =='G_MM_New_3':
        return G_MM_New_3
    elif model_name =='M_Net_Up_5_side':
        return M_Net_Up_5_side
    elif model_name =='M_Multi_1':
        return M_Multi_1
    elif model_name =='M_Net_Up_8':
        return M_Net_Up_8
    elif model_name =='M_Net_Up_9':
        return M_Net_Up_9
    elif model_name =='M_Net_Resdual':
        return M_Net_Resdual
    elif model_name =='M_Net_Resdual_Up_8':
        return M_Net_Resdual_Up_8
    elif model_name =='SR_M_Resdual':
        return SR_M_Resdual
    elif model_name =='SR_M':
        return SR_M
    elif model_name =='M_Net_Up_11':
        return M_Net_Up_11
    elif model_name =='SR_M_1':
        return SR_M_1
    elif model_name =='SR_M_only_sr':
        return SR_M_only_sr
    elif model_name =='SR_M_2':
        return SR_M_2
    elif model_name =='SR_M_3':
        return SR_M_3
    elif model_name =='SR_M_only_sr2':
        return SR_M_only_sr2
    elif model_name =='SR_M_4':
        return SR_M_4
    elif model_name =='SR_M_5':
        return SR_M_5
    elif model_name =='G_Up_temp':
        return G_Up_temp
    elif model_name =='SR_M_6':
        return SR_M_6
    elif model_name =='SR_M_7':
        return SR_M_7
    elif model_name =='SR_M_8':
        return SR_M_8
    elif model_name =='M_Net_Up_8_AT':
        return M_Net_Up_8_AT
    elif model_name =='M_Net_Up_12':
        return M_Net_Up_12
    elif model_name =='M_Net_Up_12_AT':
        return M_Net_Up_12_AT
    elif model_name =='M_Net_Up_5_AT':
        return M_Net_Up_5_AT
    elif model_name =='G_MM_1_AT':
        return G_MM_1_AT


