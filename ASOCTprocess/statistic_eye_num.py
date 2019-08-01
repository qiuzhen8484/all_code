# coding: utf-8

import os

data_path = '/data/zhangshihao/ASOCT-new/data/dataset_for_test/LGS_all/train_data'

sick_list = ['c', 'C', 'M', 'S', 's', 'H']
sick = []
normal = []
sick_num = 0
normal_num = 0
left_eye_num = 0
right_eye_num = 0
for name in os.listdir(data_path):
    eye = name.split('_')[-4]
    id = name.split('_')[0]
    if eye == 'R':
        right_eye_num += 1
    else:
        left_eye_num += 1
    if id[0] in sick_list:
        sick_num += 1
        if id not in sick:
            sick.append(id)
    else:
        normal_num += 1
        if id not in normal:
            normal.append(id)

left_eye_list = []
right_eye_list = []
for name in os.listdir(data_path):
    eye = name.split('_')[-4]
    id = name.split('_')[0]
    li = name.split('_')
    if eye == 'R':
        if id not in right_eye_list:
            right_eye_list.append(id)
    else:
        if id not in left_eye_list:
            left_eye_list.append(id)

left_txt = open('./list/Test_LGS_left_eye_list.txt', 'x')
right_txt = open('./list/Test_LGS_right_eye_list.txt', 'x')
for name in left_eye_list:
    left_txt.write(name + '\n')
for name in right_eye_list:
    right_txt.write(name + '\n')
left_txt.close()
right_txt.close()
