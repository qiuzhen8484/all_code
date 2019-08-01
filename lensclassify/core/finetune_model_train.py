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
from utils import crop_img_label
from utils import get_new_data
from utils import fit_ellipse
from utils import crop_boundry
from utils import calculate_Accuracy
from utils import get_truth
from utils import reconstruct_img
from segmentation import DecideBoundaryLeftRight
from segmentation import TransformToBinaryImage
from rayEstimation.MatchTemplateV3 import MatchTemplate

import scipy.io as scio
import time


models_list = ['UNet128',  'UNet256',       'UNet512',   'UNet1024',   'UNet512_SideOutput',  'UNet1024_SideOutput',
              'resnet_50', 'resnet_dense',  'PSP',       'dense_net',  'UNet128_deconv',      'UNet1024_deconv',
               'FPN18',    'FPN_deconv',    'My_UNet',    'M_Net',      'M_Net_deconv',       'Small',
               'VGG',       'BNM',          'BNM_1',      'BNM_2',      'BNM_3',              'Guanghui',
               'HED']

torch.manual_seed(1111)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--data_path', type=str, default='./data/dataset',
                    help='dir of the all img')
parser.add_argument('--results', type=str,  default='./visual_results',
                    help='path to save the visualization image')
parser.add_argument('--best_model', type=str,  default='70_1e-07.pth',
                    help='the pretrain model')
parser.add_argument('--finetune_model', type=str,  default='70_1e-07.pth',
                    help='the pretrain model')
parser.add_argument('--flag', type=str,  default='test',
                    help='the pretrain model')

parser.add_argument('--model_id', type=int, default=17,
                    help='the id of choice_model in models_list')
parser.add_argument('--finetune_id', type=int, default=21,
                    help='the id of choice_model in models_list')
parser.add_argument('--batch_size', type=int, default=10,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=1024,
                    help='the train img size')
parser.add_argument('--n_class', type=int, default=4,
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
train_data = os.path.join(args.data_path,str(args.n_class),'train_data')
test_data = os.path.join(args.data_path,str(args.n_class),'test_data')
label_data = os.path.join(args.data_path,str(args.n_class),'train_label')

model_name = models_list[args.model_id]
model = get_cur_model(model_name, n_classes=args.n_class, pretrain=args.pre_part)
# save_model_path = os.path.join('./models',model_name)
finetune_model_name = models_list[args.finetune_id]
finetune_model = get_cur_model(finetune_model_name, n_classes=args.n_class, pretrain=args.pre_part)

softmax_2d = nn.Softmax2d()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
if args.use_gpu:
    model.cuda()
    finetune_model.cuda()
if args.pre_all:
    model_path = os.path.join('models',model_name,args.best_model)
    model.load_state_dict(torch.load(model_path))
    finetune_model_path = os.path.join('models', finetune_model_name, args.finetune_model)
    finetune_model.load_state_dict(torch.load(finetune_model_path))

mean_loss = 100

print 'This model is %s_%s_%s_%s'%(model_name,finetune_model_name,args.img_size,args.finetune_model)
print("Begining test ")

model.eval()
finetune_model.eval()
save_test_img = True
MSE_loss_list = []
confusion = np.zeros([args.n_class, args.n_class])
test_img_list = os.listdir(test_data)
if args.flag=='train':
    test_img_list = os.listdir(train_data)

if args.hard_example_train:
    with open('./logs/hard_example.pkl') as f:
        test_img_list = pkl.load(f)

hard_example_list = []

for idx, img_path in enumerate(test_img_list):
    label_path = os.path.join(label_data, img_path)
    label = cv2.imread(label_path)
    label = cv2.resize(label, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)[:, :, :1]

    ori_img_path = os.path.join(ori_data, img_path)
    img = cv2.imread(ori_img_path, 0)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 35, 110)
    SlidingWindowSize = (20, 40)
    BinaryMap = TransformToBinaryImage(edges)
    imshape = edges.shape

    img2rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    ally = DecideBoundaryLeftRight(BinaryMap, SlidingWindowSize)
    # Draw The boundary
    BeginY = imshape[1] / 3
    newImage = img2rgb[BeginY:, ally[0]:ally[1]]
    train_img = newImage.copy()

    resize_img = cv2.resize(newImage, (args.img_size, args.img_size))
    resize_img = np.transpose(resize_img, [2, 0, 1])
    resize_img = Variable(torch.from_numpy(resize_img)).float().cuda()
    resize_img = torch.unsqueeze(resize_img, 0)

    #1 get the first output
    out = model(resize_img)
    out = torch.log(softmax_2d(out[0]))

    #2 reconstruct the image to input the 2nd Unet
    ppi = np.argmax(out.cpu().data.numpy(), 1).reshape((args.img_size, args.img_size))
    # with open('./logs/reconstruct_img.pkl','w+') as f:
    #     pkl.dump(out.cpu().data.numpy(),f)
    # assert False
    train_img, boundry_label, min_y, max_y = crop_img_label(ppi, img_path, train_img, img_size=args.img_size)
    # with open('./logs/test_data/test_img.pkl','w+') as f:
    #     pkl.dump([train_img, boundry_label, min_y, max_y],f)
    # assert False
    # train_img = reconstruct_img(ppi, train_img,img_size=args.img_size)
    # save_name = os.path.join('./logs/test_data', img_path)
    # cv2.imwrite(save_name, train_img)
    # train_img = cv2.resize(train_img, (args.img_size, args.img_size))
    # train_img = np.transpose(train_img, [2, 0, 1])
    # train_img = Variable(torch.from_numpy(train_img)).float().cuda()
    # train_img = torch.unsqueeze(train_img, 0)

    out = finetune_model(train_img)
    # loss = bce2d(out[0], boundry_label)
    # print loss.data[0]
    # out = torch.log(softmax_2d(out[0]))
    ppi = out.cpu().data.numpy().reshape((args.img_size, args.img_size))

    new_Image = ppi.astype(np.uint8)

    FinalShape = new_Image.copy()
    ori_img = cv2.imread(ori_img_path)

    ROI_img = ori_img[BeginY+min_y:BeginY+max_y, ally[0]:ally[1], :].copy()
    ROIShape = ROI_img.shape
    if save_test_img:
        pred_img = cv2.resize(new_Image.astype(np.uint8), (ROI_img.shape[1], ROI_img.shape[0]))
        pred_img = segmentation.find_boundaries(pred_img)
        pred_img = pred_img *1
        ROI_img = crop_boundry(ROI_img, pred_img)
        alpha_img = ori_img.copy()
        alpha_img[BeginY+min_y:BeginY+max_y:, ally[0]:ally[1]] = ROI_img

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
    FullImage[BeginY+min_y:BeginY+max_y, ally[0]:ally[1]] = pred_imgFinal
    # NucluesImage = MatchTemplate(FullImage)
    # NucluesImage = NucluesImage[BeginY:, ally[0]:ally[1]]
    # pred_img[pred_img == 3] = 2
    # pred_img[NucluesImage == 1] = 3
    # FullImage = np.zeros(imshape)
    # FinalShape = FinalShape * 1
    # pred_imgFinal = cv2.resize(pred_img.astype(np.uint8), (ROIShape[1], ROIShape[0]))
    # FullImage[BeginY:, ally[0]:ally[1]] = pred_imgFinal
    # cv2.imwrite('./tmp_data/%s.png' % img_path.split('.')[0], FullImage)
    tmp_img = np.zeros(imshape)
    MSE_loss = MSE_pixel_loss(tmp_img, FullImage, img_path, ally)
    print img_path,MSE_loss
    MSE_loss_list.append(MSE_loss)


    # save the best train image to compute the ray feature
    # if MSE_loss[0]<5 and MSE_loss[1]<5:
    #     save_name = './logs/best_data/%s.pkl'%(img_path.split('.')[0])
    #     with open(save_name,'w+') as f:
    #         pkl.dump([img_path,FullImage,ally, BeginY],f)
    #     print 'find the best image'
    # elif MSE_loss[0]>10 and args.flag=='train':
    #     hard_example_list.append(img_path)
    #     print 'find the hard example'

    # save_name = './logs/41/%s/%s.pkl' % (args.flag,img_path.split('.')[0])
    # with open(save_name, 'w+') as f:
    #     pkl.dump([img_path, FullImage, ally, BeginY], f)
    # print 'save the all image'

MSE_loss = np.mean(np.stack(MSE_loss_list),0)
print MSE_loss
# save_name = './logs/hard_example.pkl'
# with open(save_name, 'w+') as f:
#     pkl.dump(hard_example_list, f)
# print 'save the hard example'