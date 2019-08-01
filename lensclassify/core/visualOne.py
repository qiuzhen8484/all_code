import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
# from models import My_UNet512
from resNet_model import My_Resnet50
# from models import UnetGenerator
# from models import UNet512
from segmentation import DecideBoundaryLeftRight
from segmentation import TransformToBinaryImage
from utils import crop_boundry
# from utils import calculate_Accuracy
from utils import compute_MSE_pixel
# from utils import get_boundry_box

# from skimage import morphology
from skimage import segmentation
import argparse
import cv2
import numpy as np
import os
import scipy.io as sio
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description='PyTorch AS_OCT Demo')
parser.add_argument('--results', type=str,  default='./visual_results',
                    help='path to save the trained model')
parser.add_argument('--log', type=str,  default='./logs/log.txt',
                    help='note the best val loss')
parser.add_argument('--best_model', type=str,  default='./models/150_1e-12.pth',
                    help='the pretrain model')
parser.add_argument('--flag', type=str,  default='test',
                    help='the pretrain model')
parser.add_argument('--img_size', type=int, default=512,
                    help='the train img size')
parser.add_argument('--input_nc', type=int, default=3,
                    help='the channel of input img')
parser.add_argument('--n_class', type=int, default=2,
                    help='the channel of out img, decide the num of class')
parser.add_argument('--num_downs', type=int, default=7,
                    help='the num of the u-net block')
parser.add_argument('--gpu_num', type=int,  default=2,
                    help='the gpu id')
parser.add_argument('--img_dir', type=str,  default='/home/intern1/guanghuixu/ForDemo/Input/',
                    help='imageDir')
parser.add_argument('--out_dir', type=str,  default='/home/intern1/guanghuixu/ForDemo/Result/',
                    help='OutputDir')

args = parser.parse_args()
# model = My_UNet512([3,512,512])
model = My_Resnet50([3,512,512])
# model = UNet512([3,512,512])
model.cuda(args.gpu_num)
model.load_state_dict(torch.load(args.best_model))
print 'success the donwload the best model'
softmax_2d = nn.Softmax2d()
y_1 = []
y_2 = []

img_list = os.listdir(args.img_dir)
condition = len(img_list)
if condition > 0:
    time.sleep(2)
    for idx,img_path in enumerate(img_list):
        ori_img_path = os.path.join(args.img_dir, img_path)
        img = cv2.imread(ori_img_path, 0)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(img, 35, 110)
        SlidingWindowSize = (20, 40)
        #Get Binary Map
        BinaryMap = TransformToBinaryImage(edges)
        imshape = edges.shape
        img2rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #topx, topy = DecideBoundaryTopDown(BinaryMap_, SlidingWindowSize, 580, 1462)
        ally = DecideBoundaryLeftRight(BinaryMap, SlidingWindowSize)
        #Draw The boundary
        BeginY = imshape[1] / 3
        newImage = img2rgb[BeginY:, ally[0]:ally[1]]
        resize_img = cv2.resize(newImage, (args.img_size, args.img_size))
        resize_img = np.transpose(resize_img, [2, 0, 1])
        resize_img = Variable(torch.from_numpy(resize_img)).float().cuda(args.gpu_num)
        resize_img = torch.unsqueeze(resize_img, 0)
        out, out_2, out_3, out_4, out_5 = model(resize_img)
        pi = softmax_2d(out)
        ppi = np.argmax(pi.cpu().data.numpy(), 1).reshape((args.img_size, args.img_size))
        new_pi = pi.cpu().data.numpy().reshape((args.img_size, args.img_size),4)
        cv2.imwrite('./pixel/%s' % img_path, new_pi)
        #ppi = decode_pixel_label(ppi)
        #open and close
        new_Image = ppi.astype(np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # new_Image = cv2.morphologyEx(new_Image, cv2.MORPH_CLOSE, kernel)
        # new_Image = morphology.convex_hull_object(new_Image)
        FinalShape = new_Image.copy()

        ori_img = cv2.imread(ori_img_path)
        ROI_img = ori_img[BeginY:, ally[0]:ally[1],:].copy()
        ROIShape = ROI_img.shape
        pred_img = cv2.resize(new_Image.astype(np.uint8), (ROI_img.shape[1], ROI_img.shape[0]))
        ROI_img = crop_boundry(ROI_img, pred_img)
        alpha_img = ori_img.copy()
        alpha_img[BeginY:, ally[0]:ally[1]] = ROI_img
        ROI_img = cv2.addWeighted(alpha_img, 0.4, ori_img, 0.6, 0)

        # predict the boundary
        FullImage = np.zeros(imshape)
        FinalShape = FinalShape * 1
        pred_imgFinal = cv2.resize(FinalShape.astype(np.uint8), (ROIShape[1], ROIShape[0]))
        FullImage[BeginY:, ally[0]:ally[1]] = pred_imgFinal
        boundarys = segmentation.find_boundaries(FullImage)
        boundarys = boundarys * 1
        # boundarysxy = np.where(boundarys == 1)
        tmp_img = np.zeros(imshape)
        y_loss_1,y_loss_2 = compute_MSE_pixel(tmp_img,img_path, boundarysxy, ally)
        print y_loss_1,y_loss_2
        y_1.append(y_loss_1)
        y_2.append(y_loss_2)

        # top_idx, down_idx, center_x = get_boundry_box(boundarys)
        # with open('./MSE_pixel/boundry.txt', 'a+') as f:
        #     log_str = "%s\t%d\t%d\t%d\t%d\t%d\t" % (img_path, ally[0], top_idx, down_idx-top_idx, ally[1]-ally[0],center_x)
        #     print log_str
        # #     # str_value = str('{:d}_{:d} |  {:.2f}  | {:.2f}| {:.2f}  | {:.2f} | {:.2f}'.format(epoch, i, loss, meanIU,pixelAccuracy, meanAccuracy,classAccuracy))
        # #     str_value=str(epoch, i, loss, meanIU, pixelAccuracy, meanAccuracy, classAccuracy)
        #     f.writelines(str(log_str)+'\n')
        #
        # ori_img = cv2.imread(ori_img_path)
        # cv2.rectangle(ori_img, (ally[0],top_idx),(ally[1],down_idx), (255,0,0),10)
        # print (ally[0],top_idx),(ally[1],down_idx)
        # cv2.imwrite('./MSE_pixel/%s' % img_path, boundarys)


        # Write into mat file
        # matdir = os.path.join(args.out_dir, img_path)
        # sio.savemat(matdir,{'boundaryes':boundarysxy,'images':ROI_img})

        #delete the image
        # os.remove(ori_img_path)
        # print 'success %s and delete image' % (idx)

y_1 = np.stack(y_1)
print np.mean(y_1)
y_2 = np.stack(y_2)
print np.mean(y_2)

