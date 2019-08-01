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


def Loaddata():
    img_data = torchvision.datasets.ImageFolder("/home/intern1/qiuzhen/Works/test/datasets/trainfc", transform=transforms.Compose([transforms.Resize((256, 256)),
                                                                                   transforms.CenterCrop(224),
                                                                                   transforms.ToTensor()]))
    print(img_data.class_to_idx)
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=32, shuffle=True)
    # print(img_data.size())
    return data_loader


def testdata():
    img_data2 = torchvision.datasets.ImageFolder("/home/intern1/qiuzhen/Works/test/datasets/1", transform=transforms.Compose([transforms.Resize((256, 256)),
                                                                              transforms.CenterCrop(224),
                                                                              transforms.ToTensor()]))
    data_loader2 = torch.utils.data.DataLoader(img_data2, batch_size=32, shuffle=True)
    return data_loader2


def loadImageList(path, batchsize=32, flag='train'):
    # if flag == 'pred':
    #     image_list = []
    #     for root, dirs, files in os.walk(path):
    #         for file in files:
    #             if os.path.splitext(file)[1] == '.mat':
    #                 image_list.append(file)
    #     if (len(image_list) / batchsize - len(image_list) // batchsize) > 0:
    #         iterper_epo = len(image_list) // batchsize + 1
    #     else:
    #         iterper_epo = len(image_list) // batchsize
    #     return image_list, iterper_epo
    #
    # elif flag == 'train' or flag == 'test':
    #     image_list = []
    #     for root, dirs, files in os.walk(path):
    #         for file in files:
    #             label = scio.loadmat(os.path.join(label_path, file))['level']
    #             image_list.append({'dataname': file, 'datalabel': label})
    #
    #
    #     if (len(image_list) / batchsize - len(image_list) // batchsize) > 0:
    #         iterper_epo = len(image_list) // batchsize + 1
    #     else:
    #         iterper_epo = len(image_list) // batchsize
    #     return image_list, iterper_epo
    # with open(os.path.join(path, 'train_dict_val_unseg.pkl'), 'rb') as f:
    #     train_dict = pkl.load(f)
    # with open(os.path.join(path, 'train_dict_val_seg.pkl'), 'rb') as f1:
    #     train_dict1 = pkl.load(f1)

    with open(os.path.join(path, 'train_dict_qiu.pkl'), 'rb') as f:
        train_dict = pkl.load(f)
    if flag == 'train':
        image_list = []
        for root, dirs, files in os.walk(os.path.join(path, 'experiment_45_3', 'class_data_random')):
            for file in files:
                image_list.append(file)
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
        image_list = train_dict['val_list'] + train_dict['train_list']
        suffix = image_list[0][-4:]
        for root, dirs, files in os.walk(os.path.join(path, 'experiment_45_3', 'class_data_random')):
            for file in files:
                image_list.remove((file[:-4] + suffix))
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

                



def loadpreddata(path, iterlist):
    #load data for predict
    img_data = np.zeros((len(iterlist), 3, 224, 224), np.float)
    img_data = torch.from_numpy(img_data)
    trans = transforms.Compose([transforms.ToTensor()])
    for i in range(len(iterlist)):
        img = Image.open(os.path.join(path, iterlist[i])).convert("RGB")
        size = (224, 224)
        img2 = img.resize(size)
        img_array = trans(img2)
        img_data[i, :, :, :] = img_array
        # print(img_data[i, :, :, :])
    # img_data = torch.from_numpy(np.array(img_data))
    img_data = img_data.float()
    return img_data

def loaddata(path, iterlist):
    #load data for train and test
    data_path = path + '/train_data'
    # data_path = path + '/experiment' + '/train_data'
    # label_path = path + '/exp_45_3' + '/test_data_label_on_testmean'
    label_path = path + '/experiment_45_3' + '/train_class1'
    img_data = np.zeros((len(iterlist), 3, 224, 224), np.float)
    img_data = torch.from_numpy(img_data)
    img_label = torch.zeros(len(iterlist))
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
        # size = (256, 256)
        size = (224, 224)
        img2 = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img2 = img2.astype(np.double)
        a1 = 65535 / 256 * 150
        a2 = 65535 / 256 * 87
        img2 = np.where(img2 > a2, 255 * (img2 - a2) / (a1 - a2), 0)
        img2[img2 > 255] = 255
        img2 = img2.astype(np.uint8)

        img2 = img2 / 255
        # img2 = img2 / 65535
        # img2 = img2.astype(np.uint8)
        img2 = np.expand_dims(img2, axis=2)
        # img_array = trans(img2)
        # # # img2 = img2.astype(float)
        img_array = np.transpose(img2, (2, 0, 1))
        img_array = torch.from_numpy(img_array)
        img_data[i, 0, :, :] = img_array
        img_data[i, 1, :, :] = img_array
        img_data[i, 2, :, :] = img_array
        img_label[i] = torch.from_numpy(scio.loadmat(os.path.join(label_path, (iterlist[i][:-4] + '.mat')))['level'])

        # print(img_data[i, :, :, :])
    # img_data = torch.from_numpy(np.array(img_data))
    img_data = img_data.float()
    img_label = img_label.long()
    return img_data, img_label



# transforms.RandomHorizontalFlip()