import torch
import torch.nn as nn
import torchvision.transforms as transforms

import cv2
import numpy as np

import os
import cPickle as pkl
import argparse

from utils import get_coco_data
from utils import get_cur_color
from utils import calculate_Accuracy
from coco_process import CocoDetection

from resNet_model import resnet_test

torch.manual_seed(111)

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

parser = argparse.ArgumentParser(description='PyTorch ResNet_Test')

parser.add_argument('--coco_data', type=str, default='/home/intern1/dataset/cocoapi',
                    help='dir of the all ori img')
parser.add_argument('--results', type=str,  default='./visual_results',
                    help='path to save the trained model')
parser.add_argument('--log', type=str,  default='./logs/log.txt',
                    help='note the best val loss')
parser.add_argument('--best_model', type=str,  default='./models/20.pth',
                    help='the pretrain model')
parser.add_argument('--flag', type=str,  default='train',
                    help='the pretrain model')

parser.add_argument('--batch_size', type=int, default=10,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=512,
                    help='the train img size')
parser.add_argument('--n_class', type=int, default=81,
                    help='the channel of out img, decide the num of class, coco is 80 class')
parser.add_argument('--gpu_num', type=int,  default=[4,5],
                    help='the gpu id')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--pre_lr', type=float, default=0.000025,
                    help='pretrain model parameters learning rate, eg pretrain_resnet50')
parser.add_argument('--init_clip_max_norm', type=float, default=0.05,
                    help='pretrain model parameters learning rate, eg pretrain_resnet50')


args = parser.parse_args()
with open(os.path.join(args.coco_data,'annotations/2014_labels_dict.pkl')) as f:
    labels_dict = pkl.load(f)

model = resnet_test(pretrain=True,n_class=args.n_class)

detection = CocoDetection(root=os.path.join(args.coco_data,'coco/%s2014' %args.flag),
                          annFile=os.path.join(args.coco_data, 'annotations/instances_%s2014.json' % args.flag),
                          labels_dict=labels_dict, transform=transforms.ToTensor(), img_size=args.img_size)
import torch.nn.utils as utils
if args.init_clip_max_norm is not None:
    utils.clip_grad_norm(model.parameters(), max_norm=args.init_clip_max_norm)

# lossfunc = nn.BCELoss()
lossfunc = nn.CrossEntropyLoss()
# lossfunc = nn.NLLLoss2d()

# softmax_2d = nn.Softmax2d()
log_softmax = nn.LogSoftmax()
# optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
mask = list(map(id, model.resnet_50.parameters()))
base_params = filter(lambda p: id(p) not in mask,
                     model.parameters())
pre_params = filter(lambda p: id(p) in mask,
                     model.parameters())
optimizer = torch.optim.SGD([{'params':base_params},
                             {'params':pre_params, 'lr':args.pre_lr}],
                            lr=args.lr, momentum=0.9)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs

    model = nn.DataParallel(model)
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

try:
    model.cuda()
    # model.cuda(args.gpu_num[0])
except:
    print 'failed to get gpu'
# model.load_state_dict(torch.load(args.best_model))

train_nums = len(detection)
# train_nums = 2000
tmp_sum = 0
for epoch in range(1001):
    tmp_sum = 0
    # batch_idx = np.random.choice(train_nums,train_nums)
    batch_idx = np.arange(train_nums)
    np.random.shuffle(batch_idx)

    if epoch % 10 == 0 and epoch != 0:
        args.lr /= 10
        args.pre_lr /= 10
        optimizer = torch.optim.SGD([{'params': base_params},
                                     {'params': pre_params, 'lr': args.pre_lr}],
                                    lr=args.lr, momentum=0.9)
    if epoch !=0:
        torch.save(model.state_dict(), './models/%s.pth' % epoch)
        meanIU, pixelAccuracy, meanAccuracy, classAccuracy = calculate_Accuracy(confusion)
        # print '=========================== epoch : %s ==========================='%epoch
        # print('meanIOU: {:.2f} | pixelAccuracy: {:.2f} | meanAccuracy: {:.2f} | classAccuracy: {:.2f}  |'.format(meanIU,
        #                                                                                                          pixelAccuracy,
        #                                                                                                          meanAccuracy,
        #                                                                                                          classAccuracy))
        # with open('./logs/log.txt', 'a+') as f:
        #     log_str = "%d\t\t%.4f\t%.4f\t%.4f\t%.4f\t" % (epoch, meanIU, pixelAccuracy, meanAccuracy, classAccuracy)
        #     f.writelines(str(log_str) + '\n')
        #
        # with open('./logs/confusion.pkl', 'a+') as f:
        #     pkl.dump(confusion, f)
        #     print 'save the results success'

    confusion = np.zeros([args.n_class, args.n_class])

    for start,end in zip(xrange(0, train_nums, args.batch_size),
                         xrange(args.batch_size, train_nums+args.batch_size, args.batch_size)):
        img_idx = batch_idx[start:end]
        img, target, tmp_img, tmp_label = get_coco_data(detection, img_idx, gpu_idx=args.gpu_num)
        if len(tmp_label)==0:
            tmp_sum+=1
            continue
        model.zero_grad()
        optimizer.zero_grad()
        out = model(img)
        # out = softmax_2d(out)
        out = log_softmax(out)

        loss = lossfunc(out, target)
        print epoch, start, loss.data[0], tmp_sum


        output = out.cpu().data.numpy()
        # print output.max()
        # with open('./logs/data.pkl', 'a+') as f:
        #     tmp_data = [output,tmp_label]
        #     pkl.dump(tmp_data, f)
        #     print 'save the results success'

        # for idx,pred_label in enumerate(output):
        #     pred = np.argmax(pred_label, 0).reshape([-1])
        #     label = tmp_label[idx].reshape([-1])
        #     assert len(label)==len(pred)
        #     for i in xrange(len(label)):
        #         confusion[label[i], pred[i]] += 1
        #
        # if start>train_nums/2:
        #     meanIU, pixelAccuracy, meanAccuracy, classAccuracy = calculate_Accuracy(confusion)
        #     print('epoch_batch: {:d}_{:d} | loss: {:.2f}  | meanIOU: {:.2f} '.format(epoch, start,loss.data[0], meanIU))
        # else:
        #     print('epoch_batch: {:d}_{:d} | loss: {:.2f}  | bad_img: {:d}'.format(epoch, start,loss.data[0], tmp_sum))

        loss.backward()
        optimizer.step()

        if start % 50 == 0:
            tmp_Image = get_cur_color(tmp_label[-1])
            save_name = './tmp_label/label_%s_%s.png' % (epoch, start)
            tmp_Image = cv2.addWeighted(tmp_img, 0.3, tmp_Image, 0.7, 0)
            cv2.imwrite(save_name, tmp_Image)

            tmp_out = np.argmax(output[-1], 0)
            ppi = tmp_out.reshape((args.img_size, args.img_size))
            new_Image = ppi.astype(np.uint8)
            new_Image = np.stack([new_Image, new_Image, new_Image]).transpose([1,2,0])
            # new_Image= get_cur_color(new_Image)
            # new_Image[new_Image!=0]=255
            # new_Image = new_Image.reshape((args.img_size, args.img_size,3))
            # new_Image = cv2.addWeighted(tmp_img, 0.3, new_Image, 0.7, 0)
            # save_name = './data/dataset/train_results/label_%s_%s.png' % (epoch, start)
            # cv2.imwrite(save_name, new_Image)

            final = np.hstack([tmp_Image, new_Image])
            save_name = './visual_results/label_%s_%s.png' % (epoch, start)
            cv2.imwrite(save_name, final)



    # for i, img_idx in enumerate(batch_idx):
    #     img, label, tmp_label = get_coco_data(detection, img_idx, img_size=args.img_size, gpu_idx=args.gpu_num)
    #     model.zero_grad()
    #     out = model(img)
    #     out = torch.log(softmax_2d(out))
    #
    #     loss = lossfunc(out, label)
    #
    #     print epoch, i, loss.data[0]
    #
    #     if i%100==0:
    #         ppi = np.argmax(out.cpu().data.numpy(), 1).reshape((args.img_size, args.img_size))
    #
    #         new_Image = ppi.astype(np.uint8)
    #         save_name = './data/dataset/train_results/label_%s_%s.png' % (epoch, i)
    #         cv2.imwrite(save_name, ppi)
    #         save_name = './tmp_label/label_%s_%s.png' % (epoch, i)
    #         cv2.imwrite(save_name, tmp_label)

    # torch.save(model.state_dict(), './models/%s.pth' % epoch)
