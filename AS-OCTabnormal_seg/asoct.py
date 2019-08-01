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
    """`ASOCT_class for abnormal segmentation
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        super(AsoctDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = self.transform.transforms[0]
        self.train = train  # training set or test set
        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []

            for file in os.listdir(self.root + '/train'):
                if os.path.splitext(file)[1] == '.jpg':
                    self.train_data.append(file)
                    self.train_labels.append(file[:-3] + 'png')

        else:
            self.test_data = []
            self.test_labels = []

            for file in os.listdir(self.root + '/test'):
                if os.path.splitext(file)[1] == '.jpg':
                    self.test_data.append(file)
                    self.test_labels.append(file[:-3] + 'png')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is mask of the corresponding.
        """
        if self.train:
            img_name, target = self.train_data[index], self.train_labels[index]
            f_dir = 'train'
        else:
            img_name, target = self.test_data[index], self.test_labels[index]
            f_dir = 'test'

        img = os.path.join(self.root, f_dir, img_name)
        img = Image.open(img)
        label = os.path.join(self.root, f_dir, target)
        label = Image.open(label)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        label = torch.from_numpy(np.array(label)).long()

        return img, label, img_name

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