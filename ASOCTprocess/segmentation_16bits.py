"""
This file aims to preprocessing data stored with 16 bits which can not be presentd as picutres
author: Shihao Zhang

"""
from __future__ import division
from utils import *
import cv2
import argparse
import time
import scipy.io as scio
import os
import math
import pickle as pkl

parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--data_path', type=str, default=r'/data/zhangshihao/ASOCT-new/data/dataset_3800_16',
                    help='dir of the all img')
args = parser.parse_args()

all_img_dir = os.path.join(args.data_path, 'data')
train_label_path = os.path.join(args.data_path, 'train_label')
visual_data_path = os.path.join(args.data_path, 'visual_data')
train_data_path = os.path.join(args.data_path, 'train_data')
three_color = [(255,255,0), (255,0,0), (0,0,255)]  # yellow, red, blue
# three_color = ['#FFFF00', '#FF0000', '#0000FF']
if not os.path.isdir(train_label_path):
    os.mkdir(train_data_path)
    os.mkdir(train_label_path)
    os.mkdir(visual_data_path)

img_infor = {}
annotations = os.listdir(all_img_dir)
# mat_list = [x for x in annotations if x.endswith('.png')]
# mat_sum = len(mat_list)
# txt = open('/data/zhangshihao/ASOCT-new/data/dataset_for_test/LGS_all/LGS_test.txt', 'r')
# mat_list = []
# for line in txt.readlines():
#     mat_list.append(line[:-1])
mat_list = [x for x in annotations if x.endswith('_1.mat')]
mat_sum = len(mat_list)
# mat_list = ['M040_20180517_165617_L_CASIA2_001_003_1.mat']

# for idx, mat_name in enumerate(mat_list):
#     start = time.time()
#     # xxx.mat corresponding to xxx.jpg.
#     img_name = mat_name[:-6]+'.jpg'
#
#     # xxx_1.mat corresponding to xxx.jpg
#     # img_name = mat_name[:-6]+'.mat'
#
#     img_path = os.path.join(all_img_dir, img_name)
#
#
#     mat_path = os.path.join(all_img_dir, mat_name)
#     mat = scio.loadmat(mat_path)
#     Lens_front_x = mat['A_lf_x'][0]
#     Lens_front_y = mat['A_lf_y'][0]
#     Lens_back_x = mat['A_lb_x'][0]
#     Lens_back_y = mat['A_lb_y'][0]
#
#     Cortex_front_x = mat['A_cf_x'][0]
#     Cortex_front_y = mat['A_cf_y'][0]
#     Cortex_back_x = mat['A_cb_x'][0]
#     Cortex_back_y = mat['A_cb_y'][0]
#
#     Nucleus_front_x = mat['A_nf_x'][0]
#     Nucleus_front_y = mat['A_nf_y'][0]
#     Nucleus_back_x = mat['A_nb_x'][0]
#     Nucleus_back_y = mat['A_nb_y'][0]
#
#     img = cv2.imread(img_path, -1)
#     # img = scio.loadmat(img_path)['mm']
#     top = int(np.min(Lens_front_y) - 100)
#     left, right = int(Lens_back_x[0]), math.ceil(Lens_back_x[-1])
#     bottom0 = math.ceil(np.max(Lens_back_y))
#     if bottom0 + 100 < 1866:
#         bottom = bottom0 + 100
#     else:
#         bottom = 1866
#
#     # img = sci.loadmat(img_path)
#     # img_tmp = img['mm']
#     # img = img_tmp[:,:, np.newaxis]
#     #
#     # img_shape = img.shape
#     # top = int(600)
#     # left, right = int(Lens_back_x[0]), int(Lens_back_x[-1])
#
#     img_infor[img_name] = [left, right, top]
#
#     newImage = img[top:bottom, left:right]
#     data_savename = os.path.join(train_data_path, img_name[:-4]+'.mat')
#     visual_name = os.path.join(visual_data_path, img_name[:-4]+'.png')
#     label_savename = os.path.join(train_label_path, img_name[:-4]+'.png')
#     scio.savemat(data_savename, {'mm': newImage})
#     newImage = np.zeros((newImage.shape[0], newImage.shape[1], 3))
#
#     # np.polyfit
#     for color_id, annotation in enumerate([[Lens_front_x,Lens_front_y,Lens_back_x,Lens_back_y],
#                                         [Cortex_front_x,Cortex_front_y,Cortex_back_x,Cortex_back_y],
#                                         [Nucleus_front_x,Nucleus_front_y,Nucleus_back_x,Nucleus_back_y]]):
#         x_1,y_1,x_2,y_2 = annotation
#         front_x=x_1
#         front_y=y_1
#         back_y=y_2
#         back_y=back_y[::-1]
#         back_x=x_2
#         back_x=back_x[::-1]
#         # else:
#         #     z_1, front_x, front_y = not_csv_process(x_1, y_1, img_shape, start=left, end=right, flag=True)
#         #     z_2, back_x, back_y = not_csv_process(x_2, y_2, img_shape, start=right, end=left, flag=True)
#         #     ans_1, ans_2 = compute_intersection(z_1, z_2)
#         #     mask = front_x>ans_1
#         #     front_x = front_x[mask]
#         #     front_y = front_y[mask]
#         #     mask = front_x < ans_2
#         #     front_x = front_x[mask]
#         #     front_y = front_y[mask]
#         #
#         #     mask = back_x > ans_1
#         #     back_x = back_x[mask]
#         #     back_y = back_y[mask]
#         #     mask = back_x < ans_2
#         #     back_x = back_x[mask]
#         #     back_y = back_y[mask]
#         # lens_tmp = min(len(front_x), len(back_x))
#         # front_x = front_x[:lens_tmp]
#         # back_x = back_x[:lens_tmp]
#         # lens_tmp = min(len(front_y), len(back_y))
#         # front_y = front_y[:lens_tmp]
#         # back_y = back_y[:lens_tmp]
#
#         x = np.stack([front_x, back_x]).reshape([-1])
#         y = np.stack([front_y, back_y]).reshape([-1])
#         x = x - left
#         y = y - top
#         new_index = zip(x, y)
#         new_index = list(new_index)
#         new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
#
#         # cv2.polylines(img2,[np.int32(new_index)],False,(255,0,0),8)
#         # cv2.fillPoly(img2,[new_index],three_color[color_id])
#         cv2.fillPoly(newImage, [new_index], three_color[color_id])
#         cv2.imwrite(visual_name, newImage)
#     cv2.imwrite(visual_name, newImage)
#     # cv2.imwrite(visual_name, img2)
#     pixel_label = get_pixel_label(newImage)
#     cv2.imwrite(label_savename, pixel_label)
#     spent_time = time.time()-start
#     print('| images:%s | success:%s | time:%s |'%(mat_sum, idx+1,spent_time))
#
# with open(os.path.join(args.data_path, 'train.pkl'), 'wb+') as f:
#     pkl.dump(img_infor, f)
# print('Step 1: success all the image information')
#
#
train_dict = {}
with open(os.path.join(args.data_path, 'train_dict.pkl'), 'wb+') as f:
    train_list, val_list = split_dataset(train_label_path)
    train_dict['train_list'] = train_list
    train_dict['val_list'] = val_list
    # train_dict['test_list'] = train_list + val_list
    pkl.dump(train_dict, f)
print('Step 2: success split the dataset, train:%s, val:%s' % (len(train_list), len(val_list)))

distance_path = os.path.join(args.data_path, 'distance')
if not os.path.exists(distance_path):
    os.mkdir(distance_path)
    # distance_map(data_path=train_label_path, distance_path=distance_path)
    distance_map_new(data_path=train_label_path, distance_path=distance_path)
    print('Step 3: success generating distance_map')
else:
    print('Step 3: distance_map already exit')

