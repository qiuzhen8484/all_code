import sys
import csv
import sqlite3
import scipy.io as scio
import os
from openpyxl import load_workbook
import openpyxl
import numpy as np
# f = open(r'C:\Users\cvter\Desktop\AS-OCT\ASOCT_Label_Data\user8_transfer\201801-201803\20180112.1727702684-9675-1\20180112.1727702684-9675-1.txt', 'r')
# a = f.readlines()
# print(len(a))
# b = a[1].split()
path = r'C:\Users\cvter\Desktop\AS-OCT\ASOCT_Label_Data\user8_transfer\201804-201805'
class_savepath = r'I:\for_angle_label\user8'
path_save = r'D:\for_locate_point\user8'

for file in os.listdir(path):
    # print(file)
    flag = 1
    txt_file = os.listdir(os.path.join(path, file))[-1]
    print(txt_file)
    f = open(os.path.join(r'G:\中山眼科\now\3DV-JPG\3dv-16img3dv', file, (file + '.xpf')), 'r')
    width = int(f.readlines()[1][8:-1])
    f.close()
    f_txt = open(os.path.join(path, file, txt_file), 'r')
    rows = f_txt.readlines()
    if len(rows) == 0:
        continue
    for j in range(16):
        row = rows[j+1].split()
        left_status_1 = int(np.double(row[0]))
        right_status_1 = int(np.double(row[3]))
        if left_status_1 != 4 and right_status_1 != 4:
            gap = float(row[4]) - float(row[1])
            if gap < width:
                flag = 2130 / width
            break

    for i in range(16):
        row = rows[i+1].split()
        left_status = int(np.double(row[0]))
        right_status = int(np.double(row[3]))
        scio.savemat(os.path.join(class_savepath, (file + '_' + str(i+1) + '.mat')), {'leftstatus': left_status, 'rightstatus': right_status})
        if left_status == 4 and right_status == 4:
            scio.savemat(os.path.join(path_save, (file + '_' + str(i+1) + '.mat')), {'leftpoint': [0, 0], 'rightpoint': [0, 0]})
        elif left_status == 4:
            p1x = float(row[4]) * flag
            p1y = float(row[5])
            scio.savemat(os.path.join(path_save, (file + '_' + str(i + 1) + '.mat')), {'leftpoint': [0, 0], 'rightpoint': [p1x, p1y]})
        elif right_status == 4:
            p1x = float(row[1]) * flag
            p1y = float(row[2])
            scio.savemat(os.path.join(path_save, (file + '_' + str(i + 1) + '.mat')), {'leftpoint': [p1x, p1y], 'rightpoint': [0, 0]})
        else:
            p1x = float(row[1]) * flag
            p1y = float(row[2])
            p2x = float(row[4]) * flag
            p2y = float(row[5])
            scio.savemat(os.path.join(path_save, (file + '_' + str(i + 1) + '.mat')), {'leftpoint': [p1x, p1y], 'rightpoint': [p2x, p2y]})
    f_txt.close()


