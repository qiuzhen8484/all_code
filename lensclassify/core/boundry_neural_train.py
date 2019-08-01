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
import time
from skimage import segmentation


from utils import MSE_pixel_loss
from utils import get_cur_model
from utils import get_new_data
from utils import crop_boundry
from utils import calculate_Accuracy
from utils import crop_img_label
from utils import get_truth
from utils import bce2d
from segmentation import DecideBoundaryLeftRight
from segmentation import TransformToBinaryImage


models_list = ['UNet128',  'UNet256',       'UNet512',   'UNet1024',   'UNet512_SideOutput',  'UNet1024_SideOutput',
              'resnet_50', 'resnet_dense',  'PSP',       'dense_net',  'UNet128_deconv',      'UNet1024_deconv',
               'FPN18',    'FPN_deconv',    'My_UNet',    'M_Net',      'M_Net_deconv',       'Small',
               'VGG',       'BNM',          'BNM_1',      'BNM_2',      'BNM_3',              'Guanghui',
               'HED']

torch.manual_seed(1111)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--data_path', type=str, default='./data/dataset',
                    help='dir of the all img')
parser.add_argument('--results', type=str,  default='./visual_results',
                    help='path to save the visualization image')
parser.add_argument('--best_model', type=str,  default='pretrained.pth',
                    help='the pretrain model')
parser.add_argument('--finetune_model', type=str,  default='70_1e-07.pth',
                    help='the pretrain model')
parser.add_argument('--flag', type=str,  default='train',
                    help='the pretrain model')

parser.add_argument('--model_id', type=int, default=21,
                    help='the id of choice_model in models_list')
parser.add_argument('--epochs', type=int, default=300,
                    help='the epochs of this run')
parser.add_argument('--batch_size', type=int, default=10,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=1024,
                    help='the train img size')
parser.add_argument('--n_class', type=int, default=4,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')

parser.add_argument('--lr', type=float, default=1e-2,
                    help='initial learning rate')
parser.add_argument('--pre_lr', type=float, default=0.000025,
                    help='pretrain model parameters learning rate, eg pretrain_resnet50')
parser.add_argument('--init_clip_max_norm', type=float, default=0.05,
                    help='pretrain model parameters learning rate, eg pretrain_resnet50')

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
small_model = get_cur_model('Small',n_classes=args.n_class)
# save_model_path = os.path.join('./models',model_name)
if 'resnet' in model_name:
    params_mask = list(map(id, model.resnet.parameters()))
    base_params = filter(lambda p: id(p) not in params_mask,
                         model.parameters())
    pre_params = filter(lambda p: id(p) in params_mask,
                         model.parameters())
    optimizer = torch.optim.Adam([{'params':base_params},{'params':pre_params, 'lr':args.pre_lr}],lr=args.lr)
    # optimizer = torch.optim.SGD([{'params':base_params},
    #                              {'params':pre_params, 'lr':args.pre_lr}],
    #                             lr=args.lr, momentum=0.9)
else:
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

# lossfunc = nn.CrossEntropyLoss()
lossfunc = nn.NLLLoss2d()
softmax_2d = nn.Softmax2d()
# log_softmax = nn.LogSoftmax()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
if args.use_gpu:
    model.cuda()
    small_model.cuda()
if args.pre_all:
    # model_path = os.path.join('models',model_name,args.best_model)
    # model.load_state_dict(torch.load(model_path))
    small_model_path = os.path.join('models/Small', args.finetune_model)
    small_model.load_state_dict(torch.load(small_model_path))

img_list = os.listdir(train_data)

# the train.pkl save the boundry information of all images
with open(os.path.join(args.data_path, 'train.pkl')) as f:
    img_infor = pkl.load(f)

if args.hard_example_train:
    with open('./logs/hard_example.pkl') as f:
        img_list = pkl.load(f)
EPS = 1e-12
mean_loss = 100
for epoch in range(args.epochs):
    print 'This model is %s_%s_%s'%(model_name,args.n_class,args.img_size)
    model.train()
    small_model.eval()
    loss_list = []
    random.shuffle(img_list)

    if epoch==args.epochs*0.5 or epoch==args.epochs*0.75:
        args.lr /= 10
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for i, path in enumerate(img_list):
        label_path = os.path.join(label_data, path)
        # print label_path
        label = cv2.imread(label_path)
        label = cv2.resize(label, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)[:, :, :1]

        ori_img_path = os.path.join(ori_data, path)
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

        # 1 get the first output
        out = small_model(resize_img)
        out = torch.log(softmax_2d(out[0]))

        # 2 reconstruct the image to input the 2nd Unet
        ppi = np.argmax(out.cpu().data.numpy(), 1).reshape((args.img_size, args.img_size))
        # with open('./logs/reconstruct_img.pkl','w+') as f:
        #     pkl.dump([ppi, train_img, ally[0], ally[1], BeginY],f)
        # assert False
        train_img, boundry_label, min_y, max_y = crop_img_label(ppi, path, train_img, img_size=args.img_size)
        # print train_img.size()
        if model_name=='HED':
            out, d1, d2, d3, d4, d5 = model(train_img)
            loss = bce2d(d1, boundry_label)
            loss += bce2d(d2, boundry_label)
            loss += bce2d(d3, boundry_label)
            loss += bce2d(d4, boundry_label)
            loss += bce2d(d5, boundry_label)
            loss += bce2d(out, boundry_label)
            print('epoch_batch: {:d}_{:d} | loss: {:f}'.format(epoch, i, loss.data[0]))
        elif 'BNM_' in model_name:
            out = model(train_img)[0]
            loss = bce2d(out, boundry_label)
            print('epoch_batch: {:d}_{:d} | loss_1: {:f}'.format(epoch, i, loss.data[0]))
        ppi = out.cpu().data.numpy()
        ppi= ppi.reshape((args.img_size, args.img_size))
        ppi[ppi<0.5]=0
        ppi[ppi>=0.5]=1
    #
        loss.backward()
        optimizer.step()
        # continue
        if epoch and epoch % 10 == 0 and i %4== 0:
            torch.save(model.state_dict(), './models/%s/%s_%s.pth' % (model_name, epoch, args.lr))
            print 'success save every step model'
    #
    #     with open('./logs/%s_log.txt' % (model_name), 'a+') as f:
    #         log_str = "%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (
    #         epoch, i, loss, meanIU, pixelAccuracy, meanAccuracy, classAccuracy)
    #         f.writelines(str(log_str) + '\n')
    #
        if epoch % 4 == 0 and i %4== 0:
            left, right, top = img_infor[path]['position']
            w, h = img_infor[path]['size'][:2]

            new_Image = ppi.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            new_Image = cv2.morphologyEx(new_Image, cv2.MORPH_CLOSE, kernel)

            ori_img_path = os.path.join(ori_data, path)
            ori_img = cv2.imread(ori_img_path)
            ROI_img = ori_img[top:, left:right, :].copy()
            pred_img = cv2.resize(new_Image.astype(np.uint8), (ROI_img.shape[1], ROI_img.shape[0]),
                                  interpolation=cv2.INTER_LANCZOS4)
            ROI_img = crop_boundry(ROI_img, pred_img)
            alpha_img = ori_img.copy()
            alpha_img[top:, left:right] = ROI_img
            ROI_img = cv2.addWeighted(alpha_img, 0.4, ori_img, 0.6, 0)
            try:
                ROI_img = get_truth(ROI_img, path, [left, right], top)
            except:
                print path

            if not os.path.exists('./visual_results/%s'%model_name):
                os.mkdir('./visual_results/%s'%model_name)
            save_name = './visual_results/%s/label_%s_%s.png' % (model_name,epoch, i)
            cv2.imwrite(save_name, ROI_img)

    MSE_loss = None
    if not os.path.exists('./models/%s'%model_name):
        os.mkdir('./models/%s'%model_name)
    if args.test_every_step:
        start = time.time()
        print("Begining test ")
        model.eval()
        save_test_img = False

        MSE_loss_list = []
        confusion = np.zeros([args.n_class, args.n_class])
        test_img_list = os.listdir(test_data)
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

            resize_img = cv2.resize(newImage, (args.img_size, args.img_size))
            resize_img = np.transpose(resize_img, [2, 0, 1])
            resize_img = Variable(torch.from_numpy(resize_img)).float().cuda()
            resize_img = torch.unsqueeze(resize_img, 0)
            out = model(resize_img)
            out = torch.log(softmax_2d(out[0]))
            ppi = np.argmax(out.cpu().data.numpy(), 1).reshape((args.img_size, args.img_size))
            # tmp_gt = label.reshape([-1])
            # tmp_out = ppi.reshape([-1])
            # for i in xrange(len(tmp_gt)):
            #     confusion[tmp_gt[i], tmp_out[i]] += 1

            new_Image = ppi.astype(np.uint8)

            FinalShape = new_Image.copy()
            ori_img = cv2.imread(ori_img_path)

            ROI_img = ori_img[BeginY:, ally[0]:ally[1], :].copy()
            ROIShape = ROI_img.shape
            if save_test_img:
                pred_img = cv2.resize(new_Image.astype(np.uint8), (ROI_img.shape[1], ROI_img.shape[0]))
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
            MSE_loss_list.append(MSE_loss)
        MSE_loss = np.mean(np.stack(MSE_loss_list),0)
        if np.mean(MSE_loss)<mean_loss:
            torch.save(model.state_dict(), './models/%s/best.pth' % (model_name))
            print 'success save the best model'
            mean_loss = np.mean(MSE_loss)
        end = time.time()
        # print('epoch: {:d} | lr: {:f}  | MSE_loss: {:s}'.format(epoch, args.lr, MSE_loss))
        print('epoch: {:d} | lr: {:f}  | time: {:.2f}| MSE_loss: {:s}'.format(epoch, args.lr, end - start, MSE_loss))
        with open('./logs/test_%s_log.txt' % (model_name), 'a+') as f:
            log_str = ('epoch: {:d} | lr: {:f}  | MSE_loss: {:s}'.format(epoch, args.lr, MSE_loss))
            f.writelines(log_str + '\n')

    elif epoch and epoch%10==0:
        torch.save(model.state_dict(), './models/%s/%s_%s.pth' % (model_name, epoch, args.lr))
        print 'success save every step model'
    if epoch==args.epochs-1:
        torch.save(model.state_dict(), './models/%s/final.pth' % (model_name))
        print 'success save the final model'
