
import numpy as np
from skimage import segmentation
import os
import cv2
import pickle as pkl
import math


def CheckSegmentResult(id, image_name, segmented_imgs, img_num, loss_1_list, loss_2_list, loss_3_list, fl, flag=False):
    path = '/data/zhangshihao/ASOCT-new/data/dataset_16_LRS_final'

    loss_1, loss_2, loss_3, loss_NucleusBoundary_up, loss_NucleusBoundary_down, loss_LensBoundary_up, loss_LensBoundary_down, loss_LensBoundary2_up, \
    loss_LensBoundary2_down = compute_not_csv_loss(segmented_imgs, image_name, path, False, ployfit=False)
    loss_1_list.append(np.mean(loss_1))
    loss_2_list.append(np.mean(loss_2))
    loss_3_list.append(np.mean(loss_3))

    tmp_loss = np.mean(np.stack([loss_1_list, loss_2_list, loss_3_list]), 1)
    # print(str(1 + 1) + r'/' + str(len(self.image_list)) + ': ' + 'Nuclear: %s, Cortex: %s, Lens: %s,' % (
    #     str(np.mean(loss_1)), str(np.mean(loss_2)), str(np.mean(loss_3))))
    if id == img_num - 1:
        print('nurcles: %s ± %s' % (str(tmp_loss[0]), math.sqrt(np.var(np.stack(loss_1_list), 0))))
        print('cortex: %s ± %s' % (str(tmp_loss[1]), math.sqrt(np.var(np.stack(loss_2_list), 0))))
        print('lens: %s ± %s' % (str(tmp_loss[2]), math.sqrt(np.var(np.stack(loss_3_list), 0))))
    if flag is True:
        if id == img_num - 1:
            fl.write('nucleus: ' + str(tmp_loss[0]) + '±' + str(math.sqrt(np.var(np.stack(loss_1_list), 0))) + '\n')
            fl.write('cortex: ' + str(tmp_loss[1]) + '±' + str(math.sqrt(np.var(np.stack(loss_2_list), 0))) + '\n')
            fl.write('lens: ' + str(tmp_loss[2]) + '±' + str(math.sqrt(np.var(np.stack(loss_3_list), 0))) + '\n' + '\n')
    return loss_1_list, loss_2_list, loss_3_list


def compute_not_csv_loss(output, path, data_path, use_crop=False, ployfit=False):
    img_name = path
    # print('name  :' + path)
    # print('data path  :' + data_path)
    path = path[:-3] + 'npy'
    # path = os.path.join('/home/intern1/guanghuixu/resnet/data/new_data_1/split_distance',str(size),path)
    path = os.path.join(data_path + '/distance', path)
    # path = os.path.join(data_path + '/distance_resize256', path)
    distance_map = np.load(path)
    # distance_map = np.load(path).min(axis=0)

    if use_crop:
        with open(os.path.join(data_path, 'croped_up_and_down_dict.pkl')) as f:
            croped_up_and_down_dict = pkl.load(f)
        [h1, h2] = croped_up_and_down_dict[img_name]
        distance_map = distance_map[:, h1:h2, :]

    _, height, width = distance_map.shape
    # print output.shape,output.max()
    output = output.cpu()
    output = output.numpy()
    output = cv2.resize(output.astype(np.uint8), (width, height))
    # print output.shape

    NucleusBoundary, LensBoundary, LensBoundary2, NucleusBoundary_up, NucleusBoundary_down, LensBoundary_up, LensBoundary_down, LensBoundary2_up, LensBoundary2_down = find_3_region(output, ployfit)


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

    for i, region in enumerate([NucleusBoundary, LensBoundary, LensBoundary2]):
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

    return plt_x, plt_y
