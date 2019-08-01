import torch
import numpy as np
from dataloader import loadImageList, loaddata
from torch.autograd import Variable
import os
from evaluateSegment import *
from torchsummary import summary
import time
import cv2


def val_model(path, net, fl=None, flag=False):
    net.eval()
    batchsize = 1
    image_list, iterper_epo = loadImageList(path, batchsize=batchsize, flag='val')
    # image_list = image_list[:101]
    total = len(image_list)
    print('test_data_len:' + str(total))
    loss_1_list = []
    loss_2_list = []
    loss_3_list = []
    total_time = 0

    for i in range(iterper_epo):
        if i == (iterper_epo - 1):
            iterlist = image_list[i * batchsize:]
        else:
            iterlist = image_list[i * batchsize: (i + 1) * batchsize]
        img_data, img_label = loaddata(path, iterlist)
        # start = time.time()
        img_label = Variable(img_label.cuda())
        # img_label = Variable(img_label)
        output = net.forward(Variable(img_data.cuda()))
        avg_output = (output[0] + output[1] + output[2] + output[3] + output[4] + output[5]) / 6
        # an_image = time.time() - start
        # end = time.time()
        # an_image = end - start
        # print(str(an_image))
        # total_time += an_image
        _, predicted = torch.max(avg_output, 1)
        # cv2.imwrite(os.path.join('./out', iterlist[0]), predicted[0].cpu().numpy())
        loss_1_list, loss_2_list, loss_3_list = CheckSegmentResult(i, iterlist[0], predicted[0], iterper_epo, loss_1_list, loss_2_list, loss_3_list, fl, flag)

    tmp_loss = np.mean(np.stack([loss_1_list, loss_2_list, loss_3_list]), 1)
    return tmp_loss[2]
    # print("Accuracy of the prediction on the test dataset : %d %%" % (100 * correct / total))
    # print('on the test dataset mean_iou : ' + str(np.mean(s_iou)))
    # print('total_time: ' + str(total_time) + ' average_time: ' + str(total_time/101))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    path = '/data/zhangshihao/ASOCT-new/data/dataset_16_LRS_final'
    net = torch.load('/home/intern1/qiuzhen/Works/result_for_structure_segment/yin_U_Net_LRS_256_cv2_newdata.pkl')
    # summary(net, input_size=(3, 256, 256))
    val_model(path, net)
