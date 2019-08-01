import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import scipy.io as scio
import pickle as pkl
import cv2


def loadImageList(path, batchsize=32, flag='train'):
    with open(os.path.join(path, 'train_dict_qiu.pkl'), 'rb') as f:
        train_dict = pkl.load(f)
    if flag == 'train':
        image_list = train_dict['train_list']
        if (len(image_list) / batchsize - len(image_list) // batchsize) > 0:
            iterper_epo = len(image_list) // batchsize + 1
        else:
            iterper_epo = len(image_list) // batchsize
        return image_list, iterper_epo
    elif flag == 'test':
        image_list = train_dict['test_list']
        if (len(image_list) / batchsize - len(image_list) // batchsize) > 0:
            iterper_epo = len(image_list) // batchsize + 1
        else:
            iterper_epo = len(image_list) // batchsize
        return image_list, iterper_epo
    elif flag == 'val':
        image_list = train_dict['val_list']
        if (len(image_list) / batchsize - len(image_list) // batchsize) > 0:
            iterper_epo = len(image_list) // batchsize + 1
        else:
            iterper_epo = len(image_list) // batchsize
        return image_list, iterper_epo

    if flag == 'pred':
        image_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1] == '.mat':
                    image_list.append(file)
        if (len(image_list) / batchsize - len(image_list) // batchsize) > 0:
            iterper_epo = len(image_list) // batchsize + 1
        else:
            iterper_epo = len(image_list) // batchsize
        return image_list, iterper_epo

def loaddata(path, iterlist):
    # load data for train and test
    data_path = path + '/train_data'
    label_path = path + '/train_label'
    img_data = np.zeros((len(iterlist), 3, 256, 256), np.float)
    img_data = torch.from_numpy(img_data)
    img_label = torch.zeros((len(iterlist), 256, 256))
    # trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    trans = transforms.Compose([transforms.ToTensor()])
    for i in range(len(iterlist)):
        # img = Image.open(os.path.join(data_path, (iterlist[i][:-4] + '.jpg'))).convert("RGB")
        # size = (224, 224)
        # img2 = img.resize(size)
        # # img111 = np.array(img2)
        # img_array = trans(img2)
        # img_data[i, :, :, :] = img_array

        # 16位图
        img = scio.loadmat(os.path.join(data_path, (iterlist[i][:-4] + '.mat')))['mm']
        # size = (224, 224)
        size = (256, 256)
        img2 = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img2 = img2 / 65535
        # img2 = np.expand_dims(img2, axis=2)
        # img_array = trans(img2)
        img2 = np.expand_dims(img2, axis=2)
        # img2 = img2.astype(float)
        img_array = np.transpose(img2, (2, 0, 1))
        # img_array = img_array.astype(np.float32)
        img_array = torch.from_numpy(img_array)
        img_data[i, 0, :, :] = img_array
        img_data[i, 1, :, :] = img_array
        img_data[i, 2, :, :] = img_array

        img_label_path = os.path.join(label_path, iterlist[i])
        img_label_path = img_label_path[:-4] + '.png'
        label = Image.open(img_label_path)
        label = label.resize(size)
        label = label.split()
        img_array_label = np.asarray(label[0])
        img_array_label = torch.from_numpy(img_array_label)
        img_label[i, :, :] = img_array_label

        # print(img_data[i, :, :, :])
    # img_data = torch.from_numpy(np.array(img_data))
    img_data = img_data.float()
    img_label = img_label.long()
    return img_data, img_label

