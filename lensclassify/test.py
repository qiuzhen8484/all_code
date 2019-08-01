import torch
import numpy as np
from dataloader import loadImageList, loaddata
import model
from torch.autograd import Variable
import os
import time

def val_model(path, net):
    net.eval()
    batchsize = 32
    # batchsize = 1
    image_list, iterper_epo = loadImageList(path, batchsize=batchsize, flag='val')
    # image_list = image_list[:100]
    correct = 0
    total = len(image_list)
    print('test_data_len:' + str(total))
    distance_loss = []
    total_time = 0
    # txt = open('/home/intern1/qiuzhen/resnet34_val_result.txt', 'x')
    # txt.write('name' + ',' + 'ground_truth' + ',' + 'pred_label' + '\n')

    for i in range(iterper_epo):
        if i == (iterper_epo - 1):
            iterlist = image_list[i * batchsize:]
        else:
            iterlist = image_list[i * batchsize: (i + 1) * batchsize]
        img_data, img_label = loaddata(path, iterlist)
        img_label = Variable(img_label.cuda())
        # start = time.time()
        outputs = net.forward(Variable(img_data.cuda()))
        # an_image = time.time() - start
        # print(an_image)
        # total_time += an_image
        _, predicted = torch.max(outputs.data, 1)
        num = (predicted == img_label).sum()
        correct = correct + num
        for j in range(len(iterlist)):
            distance_loss += [abs(predicted[j]-img_label[j])]
            # txt.write(iterlist[j] + ',' + str(img_label[j]) + ',' + str(predicted[j]) + '\n')

    # print('total_time: ' + str(total_time) + ' average_time: ' + str(total_time/100))
    avg_distance = np.mean(distance_loss)
    std_distance = np.std(distance_loss, ddof=1)
    # txt.close()
    print("Accuracy of the prediction on the test dataset : %d %%" % (100 * correct / total))
    print("avg_distance: " + str(avg_distance) + " std_distance: " + str(std_distance))
    return avg_distance

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    path = '/data/zhangshihao/ASOCT-new/data/dataset_16_LRS_final'
    net = torch.load('/home/intern1/qiuzhen/Works/for_level/LRS_random_dataset_resnet34_8bit_45_3.pkl')
    avg_distance = val_model(path, net)
