import cv2
import numpy as np
from PIL import Image
import os

import torch
from torch.autograd import Variable


class CocoDetection():
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, labels_dict, transform=None, img_size=512):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.img_size = img_size
        self.labels_dict = labels_dict

    def target_transform(self, target):
        tmp_labels = {}
        for ann in target:
            label = ann['category_id']
            m = self.coco.annToMask(ann)
            if label not in tmp_labels.keys():
                tmp_labels[label] = m
            else:
                tmp_labels[label] += m

        tmp = tmp_labels.keys()

        new_label = np.zeros_like(tmp_labels[tmp[0]])
        for label, data in tmp_labels.iteritems():
            for i in xrange(data.shape[0]):
                for j in xrange(data.shape[1]):
                    if data[i, j]:
                        new_label[i, j] = self.labels_dict[label]
        new_label = cv2.resize(new_label, (self.img_size, self.img_size))
        return new_label

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        if len(target)==0:
            return [], [], [], []

        path = coco.loadImgs(img_id)[0]['file_name']

#         img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.img_size, self.img_size))
        tmp_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)

        target = self.target_transform(target)
        tmp_label = target.copy()
        target = torch.unsqueeze(torch.from_numpy(target.astype(np.int64)), 0)

        return Variable(img), Variable(target), tmp_img, tmp_label

    def __len__(self):
        return len(self.ids)
