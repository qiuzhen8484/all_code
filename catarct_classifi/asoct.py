from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch
import pickle as pkl
from torch.utils.data import Dataset



class AsoctDataset(Dataset):
    """`ASOCT-2 class
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        super(AsoctDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.npy = './data'
        # self.classes = ['N', 'C', 'M', 'S']
        self.classes = ['N', 'C']
        train_del_N = ['N276', 'n366', 'N228', 'N264', 'N088', 'N199', 'N267']
        train_del_C = ['C046', 'C067', 'co316', 'c0486', 'c0348', 'c0104', 'C049', 'c0328', 'c0257']
        val_del_N = ['N260', 'N261']
        val_del_C = ['co239', 'c091', 'C055', 'c0514', 'c0165']
        # now load the picked numpy arrays
        with open(os.path.join(self.npy, 'train_dict_N.pkl'), 'rb') as f:
            train_dict_N = pkl.load(f)
        with open(os.path.join(self.npy, 'train_dict_C.pkl'), 'rb') as f1:
            train_dict_C = pkl.load(f1)
        if self.train:
            self.train_data = []
            self.train_labels = []

            for i in range(len(self.classes)):
                label = 0 if self.classes[i] == 'N' else 1
                if self.classes[i] == 'N':
                    train_npy = train_dict_N['train_list']
                    for img_name in train_npy:
                        if img_name.split('_')[0] in train_del_N:
                            train_npy.remove(img_name)
                else:
                    train_npy = train_dict_C['train_list']
                    for img_name in train_npy:
                        if img_name.split('_')[0] in train_del_C:
                            train_npy.remove(img_name)
                ## train single picture
                # train_npy = train_npy.flatten()
                self.train_data.extend(train_npy)
                self.train_labels.extend([label] * len(train_npy))

            ## save to cpu
            # tmp = []
            # for i in range(len(self.train_data)):
            #     img_name = str(self.train_data[i], encoding="utf-8")
            #     target = self.train_labels[i]
            #     img = os.path.join(self.root, str(self.classes[target]), img_name)
            #     img = Image.open(img)
            #     tmp.append(img)
            # self.train_data = tmp

        else:
            self.test_data = []
            self.test_labels = []

            for i in range(len(self.classes)):
                label = 0 if self.classes[i] == 'N' else 1
                if self.classes[i] == 'N':
                    test_npy = train_dict_N['val_list']
                    for img_name in test_npy:
                        if img_name.split('_')[0] in val_del_N:
                            test_npy.remove(img_name)
                else:
                    test_npy = train_dict_C['val_list']
                    for img_name in test_npy:
                        if img_name.split('_')[0] in val_del_C:
                            test_npy.remove(img_name)
                # test_npy = test_npy.flatten()
                self.test_data.extend(test_npy)
                self.test_labels.extend([label] * len(test_npy))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img_name, target = self.train_data[index], self.train_labels[index]
        else:
            img_name, target = self.test_data[index], self.test_labels[index]


        # print(str(img_name,encoding="utf-8"), type(str(img_name)))
        # img_name = str(img_name, encoding="utf-8")
        img = os.path.join(self.root, img_name.split('_')[0][0].upper(), img_name)
        img = Image.open(img)
        # img = img_name
        # img = img.resize((100,233), Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str