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
from utils import get_new_data
from utils import fit_ellipse
from utils import crop_boundry
from utils import get_multi_data
from utils import calculate_Accuracy
from utils import get_truth
from segmentation import DecideBoundaryLeftRight
from segmentation import TransformToBinaryImage
from rayEstimation.MatchTemplateV3 import MatchTemplate
import scipy.io as scio
import time
from matplotlib import pyplot as plt
# models_list = ['UNet128', 'UNet256', 'UNet512', 'UNet1024', 'UNet512_SideOutput', 'UNet1024_SideOutput',
#               'resnet_50', 'resnet_dense', 'PSP', 'dense_net', 'UNet128_deconv', 'UNet1024_deconv',
#                'FPN18','FPN_deconv']
models_list = ['UNet128',  'UNet256',       'UNet512',   'UNet1024',   'UNet512_SideOutput',  'UNet1024_SideOutput',
              'resnet_50', 'resnet_dense',  'PSP',       'dense_net',  'UNet128_deconv',      'UNet1024_deconv',
               'FPN18',    'FPN_deconv',    'My_UNet',    'M_Net',      'M_Net_deconv',       'PengShuai',
               'VGG',       'BNM',          'Multi_Model']

torch.manual_seed(1111)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--data_path', type=str, default='./data/dataset',
                    help='dir of the all img')
parser.add_argument('--results', type=str,  default='./visual_results',
                    help='path to save the visualization image')
parser.add_argument('--best_model', type=str,  default='./41/70_1e-07.pth',
                    help='the pretrain model')
parser.add_argument('--flag', type=str,  default='test',
                    help='the pretrain model')

parser.add_argument('--model_id', type=int, default=3,
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


args = parser.parse_args()
ori_data = os.path.join(args.data_path,'img')
train_data = os.path.join(args.data_path,str(args.n_class),'train_data')
test_data = os.path.join(args.data_path,str(args.n_class),'test_data')
label_data = os.path.join(args.data_path,str(args.n_class),'train_label')

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
    model_path = os.path.join('models',model_name,args.best_model)
    model.load_state_dict(torch.load(model_path))

mean_loss = 100

print 'This model is %s_%s_%s'%(model_name,args.n_class,args.img_size)
print("Begining test ")

# the train.pkl save the boundry information of all images
# with open(os.path.join(args.data_path, 'train.pkl')) as f:
#     img_infor = pkl.load(f)
model.eval()
save_test_img = True
MSE_loss_list = []
ray_list = []
confusion = np.zeros([args.n_class, args.n_class])
test_img_list = os.listdir(test_data)
if args.flag=='train':
    test_img_list = os.listdir(train_data)
copy_sum = 0
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
    # BeginY = imshape[1] / 3 - 100
    BeginY = imshape[1] / 3
    newImage = img2rgb[BeginY:, ally[0]:ally[1]]

    resize_img = cv2.resize(newImage, (args.img_size, args.img_size))
    resize_img = np.transpose(resize_img, [2, 0, 1])
    resize_img = Variable(torch.from_numpy(resize_img)).float().cuda()
    resize_img = torch.unsqueeze(resize_img, 0)
    out = model(resize_img)
    # data = out[0].cpu().data.numpy()
    # data = data.squeeze()
    # data = data.transpose([1,2,0])
    # save_name = './logs/mat/%s.mat'%img_path.split('.')[0]
    # scio.savemat(save_name, {'data':[data,BeginY,ally[0],ally[1]]})
    # print 'success %s'%idx
    # with open('./logs/pkl/%s.pkl'%img_path.split('.')[0],'w+') as f:
    #     pkl.dump([data,BeginY,ally[0],ally[1]],f)'./
    # # continue
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

    FullImage = np.zeros(imshape)
    FinalShape = FinalShape * 1
    pred_imgFinal = cv2.resize(FinalShape.astype(np.uint8), (ROIShape[1], ROIShape[0]))
    FullImage[BeginY:, ally[0]:ally[1]] = pred_imgFinal
    # print FullImage.max()
    NucluesImage = MatchTemplate(FullImage)
    NucluesImage = NucluesImage[BeginY:, ally[0]:ally[1]]

    tmp_img = np.zeros(imshape)
    tmp_label = FullImage.copy()
    # with open('./logs/data/%s_side.pkl'%img_path,'w+') as f:
    #     data = [tmp_img, tmp_label, img_path, ally]
    #     pkl.dump(data, f)
    MSE_loss = MSE_pixel_loss(tmp_img, tmp_label, img_path, ally)
    print 'ori:', img_path, MSE_loss
    MSE_loss_list.append(MSE_loss)

    if save_test_img:
        pred_img = cv2.resize(new_Image.astype(np.uint8), (ROI_img.shape[1], ROI_img.shape[0]))
        tmp_img = pred_img.copy()
        tmp_ROI_img = ROI_img.copy()
        pred_img[pred_img==3]=2
        pred_img[NucluesImage==1]=3
        # with open('./logs/ray_data/%s.pkl'%img_path.split('.')[0], 'w+') as f:
        #     data = [pred_img, img_path, ally, BeginY]
        #     pkl.dump(data, f)
        # continue
        ROI_img = crop_boundry(ROI_img, pred_img)
        alpha_img = ori_img.copy()
        alpha_img[BeginY:, ally[0]:ally[1]] = ROI_img

        ROI_img = cv2.addWeighted(alpha_img, 0.4, ori_img, 0.6, 0)
        ## imshow the truth
        ROI_img = get_truth(ROI_img, img_path, ally, BeginY)

        if not os.path.exists(os.path.join(args.results,model_name)):
            os.mkdir(os.path.join(args.results,model_name))
        img_path2 = img_path[:-4] + '_ray.png'
        save_name = os.path.join(args.results,model_name, img_path2)
        cv2.imwrite(save_name, ROI_img)

        FullImage = np.zeros(imshape)
        FinalShape = FinalShape * 1
        pred_imgFinal = pred_img
        FullImage[BeginY:, ally[0]:ally[1]] = pred_imgFinal
        a = np.zeros(imshape)
        tmp_label = FullImage.copy()
        # with open('./logs/data/%s_ray.pkl'%img_path, 'w+') as f:
        #     data = [a, tmp_label, img_path, ally]
        #     pkl.dump(data, f)
        MSE_loss = MSE_pixel_loss(a, tmp_label, img_path, ally)
        print 'ray:', img_path, MSE_loss
        ray_list.append(MSE_loss)


        ROI_img = crop_boundry(tmp_ROI_img,tmp_img,)
        alpha_img = ori_img.copy()
        alpha_img[BeginY:, ally[0]:ally[1]] = ROI_img

        ROI_img = cv2.addWeighted(alpha_img, 0.4, ori_img, 0.6, 0)
        ## imshow the truth
        ROI_img = get_truth(ROI_img, img_path, ally, BeginY)
        if not os.path.exists(os.path.join(args.results, model_name)):
            os.mkdir(os.path.join(args.results, model_name))
        img_path2 = img_path[:-4] + '_ori.png'
        save_name = os.path.join(args.results, model_name, img_path2)
        cv2.imwrite(save_name, ROI_img)



    # predict the boundary
    # FullImage = np.zeros(imshape)
    # FinalShape = FinalShape * 1
    # pred_imgFinal = cv2.resize(FinalShape.astype(np.uint8), (ROIShape[1], ROIShape[0]))
    # FullImage[BeginY:, ally[0]:ally[1]] = pred_imgFinal
    # tmp_img = np.zeros(imshape)
    # tmp_label = FullImage.copy()
    # MSE_loss = MSE_pixel_loss(tmp_img, tmp_label, img_path, ally)
    #
    # print 'sideoutput:',img_path,MSE_loss
    # MSE_loss_list.append(MSE_loss)

    # # fit the ellipse
    # start = time.time()
    # save_name = os.path.join('./logs/tmp_data', model_name, img_path)
    # ROI_img = fit_ellipse(ROI_img, FullImage)
    # cv2.imwrite(save_name, ROI_img)
    # end = time.time()
    # print 'time:%s'%(end-start)

    # save the best train image to compute the ray feature
    # if MSE_loss[0]<5 and MSE_loss[1]<5:
    #save_name = './logs/test_data/%s.pkl'%(img_path.split('.')[0])
    #with open(save_name,'w+') as f:
        #pkl.dump([img_path,FullImage,ally, BeginY],f)
    # print 'find the best image'

MSE_loss = np.mean(np.stack(MSE_loss_list),0)
print 'side:',MSE_loss
MSE_loss = np.mean(np.stack(ray_list),0)
print 'ray:',MSE_loss
