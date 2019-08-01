import os
import glob
import numpy as np
from PIL import Image
from shutil import move
import csv
import scipy.io as scio
import math
import cv2
import pickle as pkl

path = r'C:\Users\cvter\Desktop\AS-OCT\ASOCT_Label_Data'
path1 = 'G:/whole/LGCboundright'
path2 = 'I:/LGS新增数据/ASOCTnormal'
path_save1 = 'F:/copy_jpg/'
path_save2 = 'F:/copy_label/'
image_list = []
# with open('/home/intern1/zhangshihao/project/ASOCT-new/data/dataset_for_test/LRS_all/train_dict.pkl', 'rb') as f:
#     train_dict = pkl.load(f)
# list = train_dict['test_list']
# print(len(list))
for root, dirs, files in os.walk(path):
    for file in files:
        if os.path.splitext(file)[0][-2:] == '_1':
            image_list.append(file)
#
# for i in range(len(image_list)):
#     # img_path = os.path.join(path, (image_list[i][:-6] + '.mat'))
#     img_path = os.path.join(path, (image_list[i][:-6] + '.jpg'))
#     save_path = os.path.join(path1, (image_list[i][:-6] + '.mat'))
#     label_path = os.path.join(path, image_list[i])
#     img = cv2.imread(img_path, -1)
#     # img = scio.loadmat(img_path)['mm']
#     A_lf_x = scio.loadmat(label_path)['A_lf_x']
#     A_lf_y = scio.loadmat(label_path)['A_lf_y']
#     A_lb_y = scio.loadmat(label_path)['A_lb_y']
#     top0 = np.min(A_lf_y)
#     if (top0 - 100) > 0:
#         top = top0 - 100
#     else:
#         top = 0
#     bottom0 = np.max(A_lb_y)
#     if (bottom0 + 100) < 1866:
#         bottom = bottom0 + 100
#     else:
#         bottom = 1866
#     left = A_lf_x[0][0]
#     right = A_lf_x[0][-1]
#     cut_img = img[int(top):math.ceil(bottom), int(left):math.ceil(right)]
#     scio.savemat(save_path, {'mm': cut_img})

for root, dirs, files in os.walk(path):
    for file in files:
        if os.path.splitext(file)[1] == '.jpg':
            for i in range(len(file)):
                if file[i] == '_':
                    ID = file[:i]
                    break
            if ID not in image_list and (ID[0] == 'N' or ID[0] == 'n'):
                image_list.append(ID)

for root, dirs, files in os.walk(path2):
    for file in files:
        if os.path.splitext(file)[1] == '.jpg':
            for i in range(len(file)):
                if file[i] == '_':
                    ID = file[:i]
                    break
            if ID not in image_list:
                image_list.append(ID)

for root, dirs, files in os.walk(path1):
    for file in files:
        if os.path.splitext(file)[1] == '.mat':
            for i in range(len(file)):
                if file[i] == '_':
                    ID = file[:i]
                    break
            if ID not in image_list:
                image_list.append(ID)
# fl = open('F:/LRS新增数据/normal_ID_for_test.txt', 'x')
# for id in image_list:
#     fl.write(id + '\n')
# fl.close()

for root1, dirs1, files1 in os.walk(path1):
    for file1 in files1:
        if os.path.splitext(file1)[1] == '.jpg':
            for j in range(len(file1)):
                if file1[j] == '_':
                    ID = file1[:j]
                    break
            if ID in image_list:
                move(os.path.join(path1, file1), path2)
                move(os.path.join(path1, (file1[:-4] + '_1.mat')), path2)

