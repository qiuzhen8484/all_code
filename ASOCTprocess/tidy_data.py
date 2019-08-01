import os
import glob
import numpy as np
from PIL import Image
from shutil import move
import csv


# 需要整理的数据路径
path = 'G:/share_data8t/201810'

# 保存统计的patientID文件的路径，需要自行创建
path_save = 'D:/ASOCTpatientcount'
path_save1 = 'D:/Otherpatientcount'

# ASOCT数据和其他数据的文件夹，需要自行创建
asoct_path = 'G:/ASOCT_Data'
other_path = 'G:/Other_Data'

data_mode = 'LRS'
sickpatient_list = []
normalpatient_list = []

# 读取ASOCT数据中健康人的patientID
file_path = 'C:/Users/cvter/Documents/WeChat Files/qz9607/Files/PatientInfo final.csv'
data = csv.reader(open(file_path, 'r'))
for id in data:
    sickpatient_list.append(id[0])

# 读取ASOCT数据中患病的人的patientID
file_path1 = 'C:/Users/cvter/Documents/WeChat Files/qz9607/Files/oct采集正常人总表.csv'
data1 = csv.reader(open(file_path1, 'r'))
for id in data1:
    normalpatient_list.append(id[0])

sick_list = []
normal_list = []
other_list = []
sick_list1 = []
normal_list1 = []
other_list1 = []

os.makedirs(os.path.join(asoct_path, 'Sick', path[-6:]))
os.makedirs(os.path.join(asoct_path, 'Normal', path[-6:]))
os.mkdir(os.path.join(other_path, path[-6:]))
sick_fl = open(os.path.join(path_save, ('sickpatient' + path[-6:] + '.txt')), 'x')
normal_fl = open(os.path.join(path_save, ('normalpatient' + path[-6:] + '.txt')), 'x')
other_fl = open(os.path.join(path_save1, ('otherpatient' + path[-6:] + '.txt')), 'x')
sick_fl.write('patient_ID' + ' ' + 'file_name' + ' left' + ' right' + '\n')
normal_fl.write('patient_ID' + ' ' + 'file_name' + ' left' + ' right' + '\n')
other_fl.write('patient_ID' + ' ' + 'file_name' + ' left' + ' right' + '\n')

for file in os.listdir(path):
    for i in range(len(file)):
        if file[i] == '_':
            filename = file[:i]
            break
    # 获取样例的3dv文件路径
    path_f = os.path.join(path, file)
    path_xpf_list = glob.glob(os.path.join(path_f, '*.3dv'))
    path_3dv = "".join(path_xpf_list)
    if len(path_3dv) > 0:
        path_xpf_l = path_3dv[:-3] + 'xpf'
        path_xpf = os.path.join(path, file, path_xpf_l)
        with open(path_xpf, "r", encoding='utf-8') as f:
            for line1 in f:
                if len(line1)>0:
                    #print(len(line1))
                    if len(line1) > 8:
                        if line1[0:8]=='AB-Scan=':
                            width = int(line1[8:-1])
                    if len(line1) > 8:
                        if line1[0:8] == 'BC-Scan=':
                            num = int(line1[8:-1])
                    if len(line1) > 6:
                        if line1[0:6] == 'Depth=':
                            Depth = int(line1[6:-1])
                    if len(line1) > 4:
                        if line1[0:4] == 'Eye=':
                            name2 = line1[4]
                    if len(line1) > 10:
                        if line1[0:10] == 'PatientID=':
                            name1 = (line1[10:-1])
                    if len(line1) > 5:
                        if line1[0:5] == 'Date=':
                            name3 = line1[5:9]+line1[10:12]+line1[13:15]
                    if len(line1) > 5:
                        if line1[0:5] == 'Time=':
                            name4 = line1[5:7]+line1[8:10]+line1[11:13]
                    if len(line1) > 23:
                        if line1[0:13] == 'ScanTypeName=':
                            if data_mode == 'LRS':
                                if line1[0:24] == 'ScanTypeName=Lens Repeat':
                                    name5 = 'LRS'
                                else:
                                    name5 = 'XXX'
                            if data_mode == 'LBO':
                                if line1[0:26] == 'ScanTypeName=Lens Biometry':
                                    name5 = 'LBO'
                                else:
                                    name5 = 'XXX'
                            if data_mode == 'LGC':
                                if line1[0:24] == 'ScanTypeName=Lens Global':
                                    name5 = 'LGC'
                                else:
                                    name5 = 'XXX'

    if name1[0] == 'n':
        name1 = 'N' + name1[1:]

    if name1 in sickpatient_list:
        if name1 not in sick_list:
            sick_list.append(name1)
            sick_list1.append({'filename': filename, 'left': 'No', 'right': 'No'})
            if name2 == 'L':
                sick_list1[-1]['left'] = 'Yes'
            else:
                sick_list1[-1]['right'] = 'Yes'
        else:
            for i in range(len(sick_list)):
                if sick_list[i] == name1:
                    k = i
                    break
            if name2 == 'L':
                sick_list1[k]['left'] = 'Yes'
            else:
                sick_list1[k]['right'] = 'Yes'
        move(os.path.join(path, file), (os.path.join(asoct_path, 'Sick', path[-6:]) + '/'))
    elif name1 in normalpatient_list:
        if name1 not in normal_list:
            normal_list.append(name1)
            normal_list1.append({'filename': filename, 'left': 'No', 'right': 'No'})
            if name2 == 'L':
                normal_list1[-1]['left'] = 'Yes'
            else:
                normal_list1[-1]['right'] = 'Yes'
        else:
            for i in range(len(normal_list)):
                if normal_list[i] == name1:
                    k = i
                    break
            if name2 == 'L':
                normal_list1[k]['left'] = 'Yes'
            else:
                normal_list1[k]['right'] = 'Yes'
        move(os.path.join(path, file), (os.path.join(asoct_path, 'Normal', path[-6:]) + '/'))
    else:
        if name1 not in other_list:
            other_list.append(name1)
            other_list1.append({'filename': filename, 'left': 'No', 'right': 'No'})
            if name2 == 'L':
                other_list1[-1]['left'] = 'Yes'
            else:
                other_list1[-1]['right'] = 'Yes'
        else:
            for i in range(len(other_list)):
                if other_list[i] == name1:
                    k = i
                    break
            if name2 == 'L':
                other_list1[k]['left'] = 'Yes'
            else:
                other_list1[k]['right'] = 'Yes'
        move(os.path.join(path, file), (os.path.join(other_path, path[-6:]) + '/'))

for i in range(len(sick_list)):
    sick_fl.write(sick_list[i] + ' ' + sick_list1[i]['filename'] + ' ' + sick_list1[i]['left'] + ' ' + sick_list1[i]['right'] + '\n')

for i in range(len(normal_list)):
    normal_fl.write(normal_list[i] + ' ' + normal_list1[i]['filename'] + ' ' + normal_list1[i]['left'] + ' ' + normal_list1[i]['right'] + '\n')

for i in range(len(other_list)):
    other_fl.write(other_list[i] + ' ' + other_list1[i]['filename'] + ' ' + other_list1[i]['left'] + ' ' + other_list1[i]['right'] + '\n')

sick_fl.close()
normal_fl.close()
other_fl.close()
