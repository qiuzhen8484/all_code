# coding: utf-8

import scipy.io as scio
import os
from PIL import Image

img_list = []
path = r'/data/zhangshihao/ASOCT-new/data/dataset_16_LGC_final/data'
for root, dirs, files in os.walk(path):
    for file in files:
        if os.path.splitext(file)[0][-2:] == '_1':
            img_list.append(file)

for name in img_list:
    img = scio.loadmat(os.path.join(path, name[:-6]+'.mat'))['mm']
    img = Image.fromarray(img)
    img = img.convert(mode='I')
    path_final = os.path.join(path, name[:-6]+'.png')
    img.save(path_final)
    os.rename(path_final, path_final[:-4] + ".jpg")