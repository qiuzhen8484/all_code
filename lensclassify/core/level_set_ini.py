import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
import numpy as np

import os
import random
import cPickle as pkl

#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from utils import get_data

from utils import MSE_pixel_loss
from utils import crop_boundry
from utils import calculate_Accuracy
from utils import get_truth
from utils import Read_ini
from utils import get_cur_model

# from visdom import Visdom
from PIL import Image
import argparse
from segmentation import DecideBoundaryLeftRight
from segmentation import TransformToBinaryImage

from LevelSets.LevelSetCNN_RNN_STN import LevelSet_CNN_RNN_STN
# from LevelSets.ShowVisdom import showImageVisdom

parser = argparse.ArgumentParser(description='Level_set')
parser.add_argument('--ini_path', type=str, default='/home/intern1/guanghuixu/resnet/scripts/Hyperparameter/lambda_2/lambda_2_0.1.ini',help='dir of the all ini path')
parser.add_argument('--gpu_idx', type=str, default=None,help='dir of the all ini path')
parser.add_argument('--finetune', type=bool, default=True,help='dir of the all ini path')
args = parser.parse_args()

conf = Read_ini(args.ini_path)

# str type=0
data_path = conf.read('Options','data_path',type=0)
ShapeTemplateName = conf.read('Options','ShapeTemplateName',type=0)
results = conf.read('Options','results',type=0)
# flag = conf.read('Options','flag',type=0)
gpu_idx = conf.read('Options','gpu_idx',type=0)
model_save_name = conf.read('Options','save_name',type=0)
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
# if args.gpu_idx:
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

# float type=1
lambda_1 = conf.read('Options','lambda_1')
lambda_2 = conf.read('Options','lambda_2')
lambda_3 = conf.read('Options','lambda_3')
e_ls = conf.read('Options','e_ls')
lambda_shape = conf.read('Options','lambda_shape')
lambda_CNN = conf.read('Options','lambda_CNN')
Lamda_RNN = conf.read('Options','Lamda_RNN')
lr = conf.read('Options','lr')
lr_decay_epoch = conf.read('Options','lr_decay_epoch')
Highe_ls = conf.read('Options','Highe_ls')

# int type=2
option_ = conf.read('Options','option_',type=2)
n_class = conf.read('Options','n_class',type=2)
InnerAreaOption = conf.read('Options','InnerAreaOption',type=2)
UseLengthItemType = conf.read('Options','UseLengthItemType',type=2)
UseHigh_Hfuntion = conf.read('Options','UseHigh_Hfuntion',type=2)
isShownVisdom = conf.read('Options','isShownVisdom',type=2)

GRU_Number = conf.read('Options','GRU_Number',type=2)
RNNEvolution = conf.read('Options','RNNEvolution',type=2)
ShapePrior = conf.read('Options','ShapePrior',type=2)
inputSize = conf.read('Options','inputSize',type=2)
gpu_num = conf.read('Options','gpu_num',type=2)
CNNEvolution = conf.read('Options','CNNEvolution',type=2)
GRU_Dimention = conf.read('Options','GRU_Dimention',type=2)
n_epochs = conf.read('Options','n_epochs',type=2)
lr_decay = conf.read('Options','lr_decay',type=2)
batch_size = conf.read('Options','batch_size',type=2)
img_size = conf.read('Options','img_size',type=2)
random_seed = conf.read('Options','random_seed',type=2)

UseHigh_Hfuntion = conf.read('Options','UseHigh_Hfuntion',type=2)

ori_data = os.path.join(data_path,'img')
train_data = os.path.join(data_path,'4','train_data')
test_data = os.path.join(data_path,'4','test_data')
label_data = os.path.join(data_path,'4','train_label')
test_flag = True

model = get_cur_model('level_set',n_class)
LevelSetModel = LevelSet_CNN_RNN_STN()

RNNLevelSetModel = 1

Options={
'InnerAreaOption':InnerAreaOption,
'UseLengthItemType':UseLengthItemType,
'UseHigh_Hfuntion':UseHigh_Hfuntion,
'isShownVisdom':isShownVisdom,
'lambda_1':lambda_1,
'lambda_2':lambda_2,
'lambda_3':lambda_3,
'lambda_shape':lambda_shape,
'lambda_CNN':lambda_CNN,
'Lamda_RNN':Lamda_RNN,
'GRU_Number':0,
'RNNEvolution':RNNEvolution,
'ShapePrior':ShapePrior,
'inputSize':(img_size,img_size),
'ShapeTemplateName':'/home/intern1/guanghuixu/resnet/shapePrior/1390_L_004_110345.pkl',
'gpu_num':0,
'CNNEvolution':CNNEvolution,
'GRU_Dimention':GRU_Dimention,
'e_ls':e_ls,
'option_':option_,
'UseHigh_Hfuntion':UseHigh_Hfuntion,
'Highe_ls':Highe_ls   # 1/1024
}

print(Options)
LevelSetModel.SetOptions(Options)

try:
    model.cuda()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
except:
    print 'failed to get gpu'

lossfunc = nn.NLLLoss2d()
softmax_2d = nn.Softmax2d()
optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

data_path = '/home/intern1/guanghuixu/resnet/data/dataset'
img_list = os.listdir(train_data)
min_loss = 100
flag = 0
with open(os.path.join(data_path,'train.pkl')) as f:
    img_infor = pkl.load(f)

if args.finetune:
    finetune_name = os.path.join('/home/intern1/guanghuixu/resnet/models/UNet512/20_1e-05.pth')
    # finetune_name = os.path.join('/home/intern1/guanghuixu/resnet/models/UNet512/60_1e-07.pth')
    finetune_dict = torch.load(finetune_name)
    model.load_state_dict(finetune_dict)
    print ('copy the weight sucessfully')

train_flag = True
if train_flag:
    for epoch in range(n_epochs):
        print ('======= %s ========'%(model_save_name)*2)
        # print('lambda_1: {:f} | lambda_2: {:f}  | lambda_3: {:f}| e_ls: {:f}  |'.format(lambda_1,lambda_2,lambda_3,e_ls))
        model.train()
        loss_list = []
        random.shuffle(img_list)
        # img_list = img_list[:2]

        # if epoch % lr_decay==0 and epoch!= 0 :
        # 	lr/=lr_decay_epoch
        # 	optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

        for i, path in enumerate(img_list):
            img, label, tmp_gt = get_data(data_path, path, img_size=img_size, n_classes=2)

            # img, label, tmp_gt = get_mini_batch_data(data_path, path,img_size=img_size,gpu_num=gpu_num)
            model.zero_grad()
            out = model(img)[0]
            # out_unet = torch.log(softmax_2d(out))
            # loss_unet = lossfunc(out_unet, label)
            # a = out.data.cpu().numpy()
            # out = torch.log(F.softmax(out))
            # label = label.view(img_size,img_size)
            # label = label.type(torch.cuda.FloatTensor)

            # Select Image and Corresponding Output, Level Set Only for Binary Classification.
            OutPut = torch.split(tensor=out, split_size_or_sections=1, dim=1)[1]
            InputImage_grey = torch.split(tensor=img, split_size_or_sections=1, dim=1)[1]
            img_ = InputImage_grey.cpu().data.numpy()
            label_ = tmp_gt
            # showImageVisdom(img_)
            # showImageVisdom(label_*127)
            # OutPut = torch.split(tensor=out, split_size=1, dim=1)[1]
            # InputImage_grey = torch.split(tensor=img, split_size=1, dim=1)[1]
            loss = LevelSetModel.Train(Image_ = InputImage_grey, OutPut = OutPut, Label_ = label)
            #loss += loss_unet
            Pre = LevelSetModel.LevelSetMask(OutPut)
            #Pre = Pre.data.cpu().numpy()
            #showImageVisdom(Pre)

            loss_list.append(loss.data[0])
            ppi = Pre
            # print ppi.max()
            loss.backward()
            tmp_gt = tmp_gt.reshape([-1])

            tmp_out = ppi.reshape([-1]).astype(np.int32)
            confusion = np.zeros([n_class, n_class])
            # print ppi.max()
            for idx in xrange(len(tmp_gt)):
                confusion[tmp_gt[idx], tmp_out[idx]] += 1
            meanIU, pixelAccuracy, meanAccuracy, classAccuracy = calculate_Accuracy(confusion)

            if epoch==n_epochs-1 and i==2:
                torch.save(model.state_dict(), '/home/intern1/guanghuixu/resnet/models/level_set/%s'%(model_save_name))
                print('success %s'%epoch)

            # print('epoch-i: {:5d,}  |  i: {:3d}  |  loss: {:5.2f}'.format(epoch, i, loss.data[0]))
            print('epoch_batch: {:d}_{:d} | loss: {:.2f}  | pixelAccuracy: {:.2f}'.format(epoch, i, loss.data[0], pixelAccuracy))
            optimizer.step()

save_test_img = True
model_name = 'level_set'
if test_flag:
    confusion = np.zeros([n_class, n_class])
    print ('======= %s========'%(model_save_name) * 2)
    # print('name: {:s} | lambda_2: {:f}  | lambda_3: {:f}| e_ls: {:f}  |'.format(model_save_name, lambda_2,lambda_3, e_ls))
    model_dict = '/home/intern1/guanghuixu/resnet/models/level_set/%s'%(model_save_name)
    model.load_state_dict(torch.load(model_dict))
    model.eval()
    MSE_loss_list = []
    test_img_list = os.listdir(test_data)
    random.shuffle(test_img_list)
    # test_img_list = test_img_list[:2]
    for idx, img_path in enumerate(test_img_list):
        label_path = os.path.join(label_data, img_path)
        left, right, BeginY = img_infor[img_path]['position']
        ally = [left, right]
        ori_img_path = os.path.join(ori_data, img_path)
        resize_img, label, tmp_gt = get_data(data_path, img_path, img_size=img_size, n_classes=2, flag='test')
        imshape = [1864, 2130]
        out = model(resize_img)[0]

        OutPut = torch.split(tensor=out, split_size_or_sections=1, dim=1)[1]
        InputImage_grey = torch.split(tensor=resize_img, split_size_or_sections=1, dim=1)[1]
        # OutPut = torch.split(tensor=out, split_size=1, dim=1)[1]
        # InputImage_grey = torch.split(tensor=resize_img, split_size=1, dim=1)[1]

        Pre = LevelSetModel.LevelSetMask(OutPut)
        # print Pre.max()
        # Pre = Pre.data.cpu().numpy()

        ResultImg = Pre
        # print ResultImg.shape
        # ppi = np.argmax(out.cpu().data.numpy(), 1).reshape((args.img_size, args.img_size))
        ppi = ResultImg.reshape((img_size, img_size))
        tmp_gt = label.reshape([-1])
        tmp_out = ppi.reshape([-1]).astype(np.int32)
        for i in xrange(len(tmp_gt)):
            confusion[tmp_gt[i], tmp_out[i]] += 1

        new_Image = ppi.astype(np.uint8)

        FinalShape = new_Image.copy()
        ori_img = cv2.imread(ori_img_path)

        ROI_img = ori_img[BeginY:, ally[0]:ally[1], :].copy()
        ROIShape = ROI_img.shape
        if save_test_img:
            pred_img = cv2.resize(new_Image.astype(np.uint8), (ROI_img.shape[1], ROI_img.shape[0]))
            # pred_img = segmentation.find_boundaries(pred_img)
            # pred_img = pred_img * 1
            ROI_img = crop_boundry(ROI_img, pred_img)
            alpha_img = ori_img.copy()
            alpha_img[BeginY:, ally[0]:ally[1]] = ROI_img

            ROI_img = cv2.addWeighted(alpha_img, 0.4, ori_img, 0.6, 0)
            ## imshow the truth
            ROI_img = get_truth(ROI_img, img_path, ally, BeginY)
            if not os.path.exists(os.path.join(results, model_name)):
                os.mkdir(os.path.join(results, model_name))
            save_name = os.path.join(results, model_name, img_path)
            cv2.imwrite(save_name, ROI_img)

        # predict the boundary
        FullImage = np.zeros(imshape)
        FinalShape = FinalShape * 1
        pred_imgFinal = cv2.resize(FinalShape.astype(np.uint8), (ROIShape[1], ROIShape[0]))
        FullImage[BeginY:, ally[0]:ally[1]] = pred_imgFinal
        # cv2.imwrite('./tmp_data/%s.png' % img_path.split('.')[0], FullImage)
        tmp_img = np.zeros(imshape)
        # print FullImage.max()
        MSE_loss = MSE_pixel_loss(tmp_img, FullImage, img_path, ally)
        print img_path, MSE_loss
        MSE_loss_list.append(MSE_loss)

    MSE_loss = np.mean(np.stack(MSE_loss_list), 0)
    meanIU, pixelAccuracy, meanAccuracy, classAccuracy = calculate_Accuracy(confusion)
    print model_save_name,MSE_loss,meanIU

    with open('/home/intern1/guanghuixu/resnet/logs/level_set/log/%s.txt'%(model_save_name.split('.')[0]), 'a+') as f:
        log_str = "%s\t%.4f\t%.4f\t%.4f\t%.4f\t" % (model_save_name, MSE_loss[0], MSE_loss[1],meanIU,pixelAccuracy)
        f.writelines(str(log_str)+'\n')
        print log_str


