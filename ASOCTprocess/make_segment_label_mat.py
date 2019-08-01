
import numpy as np
from skimage import segmentation
import os
import cv2

import scipy.io as scio
import pickle as pkl

from enumAll import DataMode

def find_3_region(boundarys, ployfit=False, ratio=2):
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
            out_up[tmp_y,tmp_x]=1
        for idx,(tmp_x,tmp_y) in enumerate(zip(DownX,DownY)):
            # in case tmp_y is out of boundary
            if tmp_y>=out_down.shape[0]:
                tmp_y=out_down.shape[0]-1
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

def findUpAndDown(boundaryMap):
    whereOne = np.where(boundaryMap == 1)
    X = whereOne[0]
    Y = whereOne[1]
    SortedY = np.sort(Y)

    # # using unique contain a problem
    # SortedY = np.unique(SortedY)

    UpX = []
    UpY = []
    DownX = []
    DownY = []

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

    return UpX,UpY,DownX,DownY

def my_ployfit(x,y,start,end,num = None, ratio=2):
    z1 = np.polyfit(x, y, ratio)
    p1 = np.poly1d(z1)
    if num is None:
        num = int(end-start+1)
    plt_x = np.linspace(start, end, num).astype(np.int32)
    plt_y = np.polyval(p1, plt_x).astype(np.int32)

    return plt_x,plt_y

def SaveMatofLable(save_path, img_dir, image_list, x, y, w, h):
    i = 0
    for id in image_list:
        top = y[i]

        base_name = os.path.basename(id)
        label_name = os.path.join(img_dir, base_name[:-4] + ".png")
        label_image = cv2.imread(label_name, 0)
        NucleusBoundary, LensBoundary, LensBoundary2, NucleusBoundary_up, NucleusBoundary_down,\
        LensBoundary_up, LensBoundary_down, LensBoundary2_up, LensBoundary2_down = find_3_region(label_image, ployfit=False)

        left = int(x[i])
        right = int(x[i]) + int(w[i])

        dataNew = os.path.join(save_path, '%s_1.mat' % base_name[:-4])
        scio.savemat(dataNew, {'A': LensBoundary,
                       'w': int(h[i]), 'top': int(top), 'left': left, 'right': right,
                       'NucleusBoundary': NucleusBoundary,
                       'LensBoundary': LensBoundary,
                       'LensBoundary2': LensBoundary2})
        i = i + 1

if __name__ == '__main__':
    data_mode = DataMode.DATA_16_LBO
    # data_mode = DataMode.DATA_16_LRS
    txt_path = "D:/left_right_top_bottom.txt"
    img_dir = "D:/11"
    save_path = "F:/LGS新增数据/normal_1"

    image_list = []
    x = []
    y = []
    w = []
    h = []
    i = 0
    with open(txt_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据

            if not lines:
                break
            content = lines.split(',')
            print(content)

            # if not os.path.exists(content[0]):
            #     continue

            image_list.append(content[0])
            x.append(content[1])
            y.append(content[2])
            w.append(content[3])
            h.append(content[4])
            i = i + 1

    SaveMatofLable(save_path, img_dir, image_list, x, y, w, h)