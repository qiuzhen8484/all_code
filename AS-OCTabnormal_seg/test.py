import torch
import numpy as np
from dataloader import loadImageList, loaddata
from torch.autograd import Variable
import os
from evaluateSegment import *
from torchsummary import summary
from metrics import Metrics
from tain_unet import iou, Dice


def val_model(path, net, fl=None, flag=False, test_loader=None):
    net.eval()
    acc_list = []
    iou_list = []
    dice_list = []

    for _, (img_data, img_label, img_name) in enumerate(test_loader):
        output = net.forward(Variable(img_data.cuda()))
        # avg_output = (output[0] + output[1] + output[2] + output[3] + output[4] + output[5]) / 6
        # _, predicted = torch.max(avg_output, 1)
        _, predicted = torch.max(output[0], 1)
        iou_list.append(iou(predicted.cpu(), Variable(img_label))[0])
        dice_list.append(Dice(predicted.cpu(), Variable(img_label))[0])
        img_label = Variable(img_label.cuda())
        correct = (predicted == img_label).sum()
        acc_list += [(int(correct) / (img_label.shape[0] * 256 * 256))]

    mean_iou_v2 = np.mean(iou_list)
    dice = np.mean(dice_list)
    acc = np.mean(acc_list)
    if flag is True:
        fl.write('test_mean_iou_v2 : ' + str(mean_iou_v2) + ' dice: ' + str(dice) + '\n')
        fl.write('test_accuracy : ' + str(acc)[:6] + '\n' + '\n')
    print("Accuracy of the prediction on the test dataset : %f %%" % (acc))
    print('on the test dataset mean_iou_v2 : ' + str(mean_iou_v2))
    print('on the test dataset dice : ' + str(dice))
    return dice


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    path = '/home/intern1/zhangshihao/project/ASOCT-new/data/dataset_16_LRS_final'
    net = torch.load('/home/intern1/qiuzhen/Works/result_for_structure_segment/U_Net_LRS_256_cv2.pkl')
    # summary(net, input_size=(3, 256, 256))
    val_model(path, net)
