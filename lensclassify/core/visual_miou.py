import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

import cv2
import numpy as np
import random
import os
import cPickle as pkl
import argparse
from skimage import segmentation

from utils import MSE_pixel_loss
from utils import get_cur_model
from utils import get_data
from utils import get_new_data
from utils import fit_ellipse
from utils import crop_boundry
from utils import calculate_Accuracy
from utils import get_truth
from segmentation import DecideBoundaryLeftRight
from segmentation import TransformToBinaryImage

import scipy.io as scio
import time


models_list = ['UNet128',  'UNet256',       'UNet512',   'UNet1024',   'UNet512_SideOutput',  'UNet1024_SideOutput',
              'resnet_50', 'resnet_dense',  'PSP',       'dense_net',  'UNet128_deconv',      'UNet1024_deconv',
               'FPN18',    'FPN_deconv',    'My_UNet',    'M_Net',      'M_Net_deconv',       'Small',
               'VGG',       'BNM',          'Guanghui',   'HED']

torch.manual_seed(1111)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--data_path', type=str, default='./data/dataset',
                    help='dir of the all img')
parser.add_argument('--results', type=str,  default='./visual_results',
                    help='path to save the visualization image')
parser.add_argument('--best_model', type=str,  default='20_2.5e-06.pth',
                    help='the pretrain model')
parser.add_argument('--flag', type=str,  default='test',
                    help='the pretrain model')

parser.add_argument('--model_id', type=int, default=2,
                    help='the id of choice_model in models_list')
parser.add_argument('--batch_size', type=int, default=10,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=512,
                    help='the train img size')
parser.add_argument('--n_class', type=int, default=2,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')


parser.add_argument('--use_gpu', type=bool, default=True,
                    help='dir of the all ori img')
parser.add_argument('--test_every_step', type=bool, default=False,
                    help='test after every train step')
parser.add_argument('--pre_all', type=bool, default=True,
                    help='pretrain the whole model')
parser.add_argument('--pre_part', type=bool, default=True,
                    help='pretrain the pytorch pretrain_model, eg resnet50 and resnet18, used in resnet_50 and resnet_dense')
parser.add_argument('--hard_example_train', type=bool, default=False,
                    help='only train the hard example')


args = parser.parse_args()
ori_data = os.path.join(args.data_path,'img')
# train_data = os.path.join(args.data_path,str(args.n_class),'train_data')
# test_data = os.path.join(args.data_path,str(args.n_class),'test_data')
# label_data = os.path.join(args.data_path,str(args.n_class),'train_label')
train_data = os.path.join(args.data_path,'4','train_data')
test_data = os.path.join(args.data_path,'4','test_data')
label_data = os.path.join(args.data_path,'4','train_label')

model_name = models_list[args.model_id]
model = get_cur_model(model_name, n_classes=args.n_class, pretrain=args.pre_part)
# save_model_path = os.path.join('./models',model_name)


softmax_2d = nn.Softmax2d()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
if args.use_gpu:
    model.cuda()
if args.pre_all:
    # print model.state_dict().keys()
    model_path = os.path.join('models',model_name,args.best_model)
    # print torch.load(model_path).keys()
    model.load_state_dict(torch.load(model_path))

mean_loss = 100

print 'This model is %s_%s_%s_%s'%(model_name,args.n_class,args.img_size,args.best_model)
print("Begining test ")

model.eval()
save_test_img = True
MSE_loss_list = []
confusion = np.zeros([args.n_class, args.n_class])
test_img_list = os.listdir(test_data)
if args.flag=='train':
    test_img_list = os.listdir(train_data)

if args.hard_example_train:
    with open('./logs/hard_example.pkl') as f:
        test_img_list = pkl.load(f)

with open(os.path.join(args.data_path,'train.pkl')) as f:
    img_infor = pkl.load(f)

hard_example_list = []
confusion = np.zeros([args.n_class, args.n_class])
# test_img_list = test_img_list[:2]
for idx, img_path in enumerate(test_img_list):
    label_path = os.path.join(label_data, img_path)
    left, right, BeginY = img_infor[img_path]['position']
    ally = [left, right]
    ori_img_path = os.path.join(ori_data, img_path)
    resize_img, label, tmp_gt = get_data(args.data_path, img_path, img_size=args.img_size, n_classes=2, flag=args.flag)
    imshape = [1864, 2130]
    if model_name=='HED' or 'BNM_' in model_name:
        out = model(resize_img)
        out = out[0]
    else:
        out = model(resize_img)
        out = torch.log(softmax_2d(out[0]))

    ppi = np.argmax(out.cpu().data.numpy(), 1).reshape((args.img_size, args.img_size))
    tmp_gt = label.reshape([-1])
    tmp_out = ppi.reshape([-1])
    for i in xrange(len(tmp_gt)):
        confusion[tmp_gt[i], tmp_out[i]] += 1

    new_Image = ppi.astype(np.uint8)

    FinalShape = new_Image.copy()
    ori_img = cv2.imread(ori_img_path)

    ROI_img = ori_img[BeginY:, ally[0]:ally[1], :].copy()
    ROIShape = ROI_img.shape
    if save_test_img:
        pred_img = cv2.resize(new_Image.astype(np.uint8), (ROI_img.shape[1], ROI_img.shape[0]))
        pred_img = segmentation.find_boundaries(pred_img)
        pred_img = pred_img * 1
        ROI_img = crop_boundry(ROI_img, pred_img)
        alpha_img = ori_img.copy()
        alpha_img[BeginY:, ally[0]:ally[1]] = ROI_img

        ROI_img = cv2.addWeighted(alpha_img, 0.4, ori_img, 0.6, 0)
        ## imshow the truth
        ROI_img = get_truth(ROI_img, img_path, ally, BeginY)
        if not os.path.exists(os.path.join(args.results,model_name)):
            os.mkdir(os.path.join(args.results,model_name))
        save_name = os.path.join(args.results,model_name, img_path)
        cv2.imwrite(save_name, ROI_img)

    # predict the boundary
    FullImage = np.zeros(imshape)
    FinalShape = FinalShape * 1
    pred_imgFinal = cv2.resize(FinalShape.astype(np.uint8), (ROIShape[1], ROIShape[0]))
    FullImage[BeginY:, ally[0]:ally[1]] = pred_imgFinal
    # cv2.imwrite('./tmp_data/%s.png' % img_path.split('.')[0], FullImage)
    tmp_img = np.zeros(imshape)
    MSE_loss = MSE_pixel_loss(tmp_img, FullImage, img_path, ally)
    print img_path,MSE_loss
    MSE_loss_list.append(MSE_loss)

MSE_loss = np.mean(np.stack(MSE_loss_list),0)
MSE_con = np.var(np.stack(MSE_loss_list),0)
meanIU, pixelAccuracy, meanAccuracy, classAccuracy = calculate_Accuracy(confusion)
print MSE_loss,meanIU,pixelAccuracy,meanAccuracy,classAccuracy
print MSE_con
# save_name = './logs/hard_example.pkl'
# with open(save_name, 'w+') as f:
#     pkl.dump(hard_example_list, f)
# print 'save the hard example'