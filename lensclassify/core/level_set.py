import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
import numpy as np

import os
import random
import cPickle as pkl

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from utils import get_data

from utils import MSE_pixel_loss
from utils import crop_boundry
from utils import calculate_Accuracy
from utils import get_truth

# from visdom import Visdom
from PIL import Image
from utils import choice_leves_set_shape
import argparse
from segmentation import DecideBoundaryLeftRight
from segmentation import TransformToBinaryImage
from LevelSetRNNmodel import LevelSet_CNN_RNN_STN, showImageVisdom

# viz = Visdom()

parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--data_path', type=str, default='./data/dataset',
                    help='dir of the all img')
parser.add_argument('--results', type=str,  default='./visual_results',
                    help='path to save the visualization image')
parser.add_argument('--best_model', type=str,  default='pretrained.pth',
                    help='the pretrain model')
parser.add_argument('--flag', type=str,  default='train',
                    help='the pretrain model')

parser.add_argument('--model_id', type=int, default=5,
                    help='the id of choice_model in models_list')
parser.add_argument('--epochs', type=int, default=300,
                    help='the epochs of this run')
parser.add_argument('--batch_size', type=int, default=10,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=512,
                    help='the train img size')
parser.add_argument('--n_class', type=int, default=2,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--seed', type=int, default=2321,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--pre_lr', type=float, default=0.000025,
                    help='pretrain model parameters learning rate, eg pretrain_resnet50')
parser.add_argument('--init_clip_max_norm', type=float, default=0.05,
                    help='pretrain model parameters learning rate, eg pretrain_resnet50')


parser.add_argument('--a_1', type=float, default=1.0,
                    help='pretrain model parameters learning rate, eg pretrain_resnet50')
parser.add_argument('--lambda_2', type=float, default=0.5,
                    help='pretrain model parameters learning rate, eg pretrain_resnet50')
parser.add_argument('--lambda_3', type=float, default=0.0,
                    help='pretrain model parameters learning rate, eg pretrain_resnet50')
parser.add_argument('--e_ls', type=float, default=1.0/32.0,
                    help='pretrain model parameters learning rate, eg pretrain_resnet50')

parser.add_argument('--use_gpu', type=bool, default=True,
                    help='dir of the all ori img')
parser.add_argument('--gpu_num', type=int, default=0,
                    help='the gpu_num to use')


parser.add_argument('--patch', type=bool, default=False,
                    help='trained on the patch')
parser.add_argument('--finetune', type=bool, default=True,
                    help='trained on the patch')
parser.add_argument('--test_every_step', type=bool, default=False,
                    help='test after every train step')
parser.add_argument('--pre_all', type=bool, default=False,
                    help='pretrain the whole model')
parser.add_argument('--pre_part', type=bool, default=True,
                    help='pretrain the pytorch pretrain_model, eg resnet50 and resnet18, used in resnet_50 and resnet_dense')
parser.add_argument('--hard_example_train', type=bool, default=False,
                    help='only train the hard example')


args = parser.parse_args()
ori_data = os.path.join(args.data_path,'img')
train_data = os.path.join(args.data_path,'4','train_data')
test_data = os.path.join(args.data_path,'4','test_data')
label_data = os.path.join(args.data_path,'4','train_label')

a_1 = args.a_1
lambda_2 = args.lambda_2
lambda_3 = args.lambda_3
e_ls = args.e_ls
option_ = 2
test_flag = True
n_class = args.n_class
gpu_num = args.gpu_num

from utils import get_cur_model
#a = TestHeavisideFunction()
model = get_cur_model('level_set',2)
LevelSetModel = LevelSet_CNN_RNN_STN()
Options={
'InnerAreaOption':2,
'UseLengthItemType':1,
'isShownVisdom':0,
'lambda_1':a_1,
'lambda_2':lambda_2,
'lambda_3':lambda_3,
'RNNEvolution':0,
'ShapePrior':0,
'inputSize':(args.img_size,args.img_size),
'ShapeTemplateName':'/home/intern1/yinpengshuai/levelset/ShapePrior/1390_L_004_110345.pkl',
'gpu_num':gpu_num,
'CNNEvolution':1
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
lr = 0.000025
optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

data_path = './data/dataset'
img_list = os.listdir(train_data)
min_loss = 100
flag = 0
img_size = args.img_size
with open(os.path.join(data_path,'train.pkl')) as f:
    img_infor = pkl.load(f)

if args.finetune:
    finetune_name = os.path.join('models/UNet512/60_1e-07.pth')
    finetune_dict = torch.load(finetune_name)
    model.load_state_dict(finetune_dict)
    # model_dict = model.state_dict()
    # # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # # # 3. load the new state dict
    # model.load_state_dict(model_dict)
    print ('copy the weight sucessfully')

train_flag = True
n_epochs = 100
if train_flag:
    for epoch in range(n_epochs):
        print ('===============')*10
        print('lambda_1: {:f} | lambda_2: {:f}  | lambda_3: {:f}| e_ls: {:f}  |'.format(args.a_1,args.lambda_2,args.lambda_3,args.e_ls))
        model.train()
        loss_list = []
        random.shuffle(img_list)
        # img_list = img_list[:2]

        if epoch % 20==0 and epoch!= 0 :
        	lr/=10
        	optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

        for i, path in enumerate(img_list):
            img, label, tmp_gt = get_data(data_path, path, img_size=img_size, n_classes=2)

            # img, label, tmp_gt = get_mini_batch_data(data_path, path,img_size=img_size,gpu_num=gpu_num)
            model.zero_grad()
            out = model(img)[0]
            # out_unet = torch.log(softmax_2d(out))
            # shape = choice_leves_set_shape(out_unet.data.cpu().numpy())

            # with open('./logs/unet_output.pkl','w+') as f:
            #     data = out_unet.data.cpu().numpy()
            #     pkl.dump(data,f)
            # assert False
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
            showImageVisdom(img_)
            showImageVisdom(label_*127)
            # OutPut = torch.split(tensor=out, split_size=1, dim=1)[1]
            # InputImage_grey = torch.split(tensor=img, split_size=1, dim=1)[1]

            loss = LevelSetModel.LevelSetLoss(Image_=InputImage_grey, OutPut_FeatureMap = OutPut, LabelMap = label)
            #loss += loss_unet
            Pre = LevelSetModel.LevelSetMask(OutPut)
            #Pre = Pre.data.cpu().numpy()
            showImageVisdom(Pre)

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
                torch.save(model.state_dict(), './models/level_set/%s_%s_%s_%s.pth'%(str(args.a_1),str(args.lambda_2),str(args.lambda_3),str(args.e_ls)))
                print('success %s'%epoch)

            # print('epoch-i: {:5d,}  |  i: {:3d}  |  loss: {:5.2f}'.format(epoch, i, loss.data[0]))
            print('epoch_batch: {:d}_{:d} | loss: {:.2f}  | pixelAccuracy: {:.2f}'.format(epoch, i, loss.data[0], pixelAccuracy))
            optimizer.step()

save_test_img = True
model_name = 'level_set'
if test_flag:
    print ('======= val ========') * 5
    print('lambda_1: {:f} | lambda_2: {:f}  | lambda_3: {:f}| e_ls: {:f}  |'.format(args.a_1, args.lambda_2,
                                                                                    args.lambda_3, args.e_ls))
    model_dict = './models/level_set/%s_%s_%s_%s.pth'%(str(args.a_1),str(args.lambda_2),str(args.lambda_3),str(args.e_ls))
    model.load_state_dict(torch.load(model_dict))
    model.eval()
    MSE_loss_list = []
    test_img_list = os.listdir(test_data)
    random.shuffle(test_img_list)
    # test_img_list = test_img_list[:2]
    confusion = np.zeros([args.n_class, args.n_class])
    for idx, img_path in enumerate(test_img_list):
        label_path = os.path.join(label_data, img_path)
        left, right, BeginY = img_infor[img_path]['position']
        ally = [left, right]
        ori_img_path = os.path.join(ori_data, img_path)
        resize_img, label, tmp_gt = get_data(data_path, img_path, img_size=img_size, n_classes=2, flag='test_data')
        imshape = [1864, 2130]
        out = model(resize_img)[0]
        # out = softmax_2d(out[0])

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
        ppi = ResultImg.reshape((args.img_size, args.img_size))
        tmp_out = ppi.reshape([-1]).astype(np.int32)
        for idx in xrange(len(tmp_gt)):
            confusion[tmp_gt[idx], tmp_out[idx]] += 1

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
            if not os.path.exists(os.path.join(args.results, model_name)):
                os.mkdir(os.path.join(args.results, model_name))
            save_name = os.path.join(args.results, model_name, img_path)
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
    print MSE_loss,meanIU,pixelAccuracy,meanAccuracy,classAccuracy
    MSE_con = np.var(np.stack(MSE_loss_list), 0)
    print MSE_con

    # with open('./logs/level_set/log/%s_%s_%s_%s.txt'%(str(args.a_1),str(args.lambda_2),str(args.lambda_3),str(args.e_ls)), 'a+') as f:
    #     log_str = "%d\t%d\t%.4f\t%.4f\t" % (epoch, i, MSE_loss[0], MSE_loss[1])
    #     f.writelines(str(log_str)+'\n')
    #     print log_str


