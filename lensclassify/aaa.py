# -*- coding: utf-8 -*-

import numpy as np
import os
from PIL import Image
import math
import json
import scipy.io as scio
import sqlite3


# path_save = r'D:\for_locate_point\user5-8_final_label_0703'
# image_list = []
# file_list = ['D:/for_locate_point/user5_1', 'D:/for_locate_point/user6_1', 'D:/for_locate_point/user7_1', 'D:/for_locate_point/user8_1']
#
# for i in range(4):
#     for root, dirs, files in os.walk(file_list[i]):
#         for file in files:
#             if file not in image_list:
#                 image_list.append(file)
#
# for i in range(len(image_list)):
#     left_point_x = []
#     left_point_y = []
#     right_point_x = []
#     right_point_y = []
#     if os.path.exists(os.path.join(file_list[0], image_list[i])):
#         a = scio.loadmat(os.path.join(file_list[0], image_list[i]))
#         if a['leftpoint'][0][0] != 0:
#             left_point_x.append(a['leftpoint'][0][0])
#             left_point_y.append(a['leftpoint'][0][1])
#         if a['rightpoint'][0][0] != 0:
#             right_point_x.append(a['rightpoint'][0][0])
#             right_point_y.append(a['rightpoint'][0][1])
#     if os.path.exists(os.path.join(file_list[1], image_list[i])):
#         a = scio.loadmat(os.path.join(file_list[1], image_list[i]))
#         if a['leftpoint'][0][0] != 0:
#             left_point_x.append(a['leftpoint'][0][0])
#             left_point_y.append(a['leftpoint'][0][1])
#         if a['rightpoint'][0][0] != 0:
#             right_point_x.append(a['rightpoint'][0][0])
#             right_point_y.append(a['rightpoint'][0][1])
#     if os.path.exists(os.path.join(file_list[2], image_list[i])):
#         a = scio.loadmat(os.path.join(file_list[2], image_list[i]))
#         if a['leftpoint'][0][0] != 0:
#             left_point_x.append(a['leftpoint'][0][0])
#             left_point_y.append(a['leftpoint'][0][1])
#         if a['rightpoint'][0][0] != 0:
#             right_point_x.append(a['rightpoint'][0][0])
#             right_point_y.append(a['rightpoint'][0][1])
#     if os.path.exists(os.path.join(file_list[3], image_list[i])):
#         a = scio.loadmat(os.path.join(file_list[3], image_list[i]))
#         if a['leftpoint'][0][0] != 0:
#             left_point_x.append(a['leftpoint'][0][0])
#             left_point_y.append(a['leftpoint'][0][1])
#         if a['rightpoint'][0][0] != 0:
#             right_point_x.append(a['rightpoint'][0][0])
#             right_point_y.append(a['rightpoint'][0][1])
#
#     if len(left_point_x) == 0 and len(right_point_x) == 0:
#         scio.savemat(os.path.join(path_save, image_list[i]), {'leftpoint': [0, 0], 'rightpoint': [0, 0]})
#     elif len(left_point_x) == 0:
#         rightpoint = [np.mean(right_point_x), np.mean(right_point_y)]
#         scio.savemat(os.path.join(path_save, image_list[i]), {'leftpoint': [0, 0], 'rightpoint': rightpoint})
#     elif len(right_point_x) == 0:
#         leftpoint = [np.mean(left_point_x), np.mean(left_point_y)]
#         scio.savemat(os.path.join(path_save, image_list[i]), {'leftpoint': leftpoint, 'rightpoint': [0, 0]})
#     else:
#         leftpoint = [np.mean(left_point_x), np.mean(left_point_y)]
#         rightpoint = [np.mean(right_point_x), np.mean(right_point_y)]
#         scio.savemat(os.path.join(path_save, image_list[i]), {'leftpoint': leftpoint, 'rightpoint': rightpoint})


path_save = r'C:\Users\cvter\Desktop\AS-OCT\relabel50_0726\point'
class_savepath = r'C:\Users\cvter\Desktop\AS-OCT\relabel50_0726\angle'

conn = sqlite3.connect(r'D:\skeptical point\Data\label.db')
ret = conn.execute("select * from asoct_label")    #获取该表所有元素
ret2 = conn.execute("select * from asoct_update_label")
conn.commit()
rows1 = ret.fetchall()
rows2 = ret2.fetchall()

for row in rows1:
    # print(row[9]) #这里就是获取去除来的每行的第2个元素的内容，row[0]则是第一个
    label_name = row[3][:-4] + '_' + str(row[4]) + '.mat'
    # print(label_name)
    # if label_name == '20180521.1727702684-13959-1_9.mat' or label_name == '20180226.1727702684-10743-1_2.mat' or label_name== '20170731.1727702684-2706-1_4.mat'\
    #         or label_name == '20171214.1727702684-7727-1_15.mat':
    #     continue
    label = row[9]
    label = json.loads(label)
    left_flag = int(label['left_radio_value'])
    right_flag = int(label['right_radio_value'])
    scio.savemat(os.path.join(class_savepath, label_name), {'leftstatus': left_flag, 'rightstatus': right_flag})
    radio = float(row[7])
    if left_flag == 4 and right_flag == 4:
        scio.savemat(os.path.join(path_save, label_name), {'leftpoint': [0, 0], 'rightpoint': [0, 0]})
    elif left_flag == 4:
        label1 = label['label_data'][0]['data']
        label1 = json.loads(label1)
        p1x = (label1['left'] + 0.5 * label1['width']) / radio
        p1y = (label1['top'] + 0.5 * label1['height']) / radio
        scio.savemat(os.path.join(path_save, label_name), {'leftpoint': [0, 0], 'rightpoint': [p1x, p1y]})
    elif right_flag == 4:
        label1 = label['label_data'][0]['data']
        label1 = json.loads(label1)
        p1x = (label1['left'] + 0.5 * label1['width']) / radio
        p1y = (label1['top'] + 0.5 * label1['height']) / radio
        scio.savemat(os.path.join(path_save, label_name), {'leftpoint': [p1x, p1y], 'rightpoint': [0, 0]})
    else:
        label1 = label['label_data'][0]['data']
        label1 = json.loads(label1)
        p1x = (label1['left'] + 0.5 * label1['width']) / radio
        p1y = (label1['top'] + 0.5 * label1['height']) / radio
        label2 = label['label_data'][1]['data']
        label2 = json.loads(label2)
        p2x = (label2['left'] + 0.5 * label2['width']) / radio
        p2y = (label2['top'] + 0.5 * label2['height']) / radio
        if p1x < p2x:
            scio.savemat(os.path.join(path_save, label_name), {'leftpoint': [p1x, p1y], 'rightpoint': [p2x, p2y]})
        else:
            scio.savemat(os.path.join(path_save, label_name), {'leftpoint': [p2x, p2y], 'rightpoint': [p1x, p1y]})

for row in rows2:
    # print(row[9]) #这里就是获取去除来的每行的第2个元素的内容，row[0]则是第一个
    label_name = row[3][:-4] + '_' + str(row[4]) + '.mat'
    label = row[9]
    label = json.loads(label)
    left_flag = int(label['left_radio_value'])
    right_flag = int(label['right_radio_value'])
    scio.savemat(os.path.join(class_savepath, label_name), {'leftstatus': left_flag, 'rightstatus': right_flag})
    radio = float(row[7])
    if left_flag == 4 and right_flag == 4:
        scio.savemat(os.path.join(path_save, label_name), {'leftpoint': [0, 0], 'rightpoint': [0, 0]})
    elif left_flag == 4:
        label1 = label['label_data'][0]['data']
        label1 = json.loads(label1)
        p1x = (label1['left'] + 0.5 * label1['width']) / radio
        p1y = (label1['top'] + 0.5 * label1['height']) / radio
        scio.savemat(os.path.join(path_save, label_name), {'leftpoint': [0, 0], 'rightpoint': [p1x, p1y]})
    elif right_flag == 4:
        label1 = label['label_data'][0]['data']
        label1 = json.loads(label1)
        p1x = (label1['left'] + 0.5 * label1['width']) / radio
        p1y = (label1['top'] + 0.5 * label1['height']) / radio
        scio.savemat(os.path.join(path_save, label_name), {'leftpoint': [p1x, p1y], 'rightpoint': [0, 0]})
    else:
        label1 = label['label_data'][0]['data']
        label1 = json.loads(label1)
        p1x = (label1['left'] + 0.5 * label1['width']) / radio
        p1y = (label1['top'] + 0.5 * label1['height']) / radio
        label2 = label['label_data'][1]['data']
        label2 = json.loads(label2)
        p2x = (label2['left'] + 0.5 * label2['width']) / radio
        p2y = (label2['top'] + 0.5 * label2['height']) / radio
        if p1x < p2x:
            scio.savemat(os.path.join(path_save, label_name), {'leftpoint': [p1x, p1y], 'rightpoint': [p2x, p2y]})
        else:
            scio.savemat(os.path.join(path_save, label_name), {'leftpoint': [p2x, p2y], 'rightpoint': [p1x, p1y]})

conn.close()

