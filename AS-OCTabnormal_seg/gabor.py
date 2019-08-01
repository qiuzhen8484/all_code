import cv2
import numpy as np
import pylab as pl
from PIL import Image

#构建Gabor滤波器
def build_filters():
    filters = []
    ksize = [5, 15, 25] #gabor尺度 6个
    lamda = 5  #np.pi * 2.0 # 波长

    for k in range(3):
        kern = cv2.getGaborKernel((ksize[k], ksize[k]), 10, np.pi/2, lamda, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()  # 1.5*kern.sum()
        filters.append(kern)
    return filters

# 滤波过程
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    # fimg = cv2.filter2D(img, cv2.CV_8UC3, filters)
    # np.maximum(accum, fimg, accum)
    return accum


# 特征图生成并显示
def getGabor(img, filters):
    image = Image.open(img)
    img_ndarray = np.asarray(image)
    res = []  # 滤波结果
    for i in range(len(filters)):
        res1 = process(img_ndarray, filters[i])
        res.append(np.asarray(res1))

    return res

