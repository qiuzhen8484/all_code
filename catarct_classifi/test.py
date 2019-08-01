import torch
import numpy as np
import model
from torch.autograd import Variable
import os
import torchvision.transforms as transforms
from asoct import AsoctDataset
import time
from sklearn.metrics import auc, confusion_matrix
from sklearn import metrics

def val_model(net, test_loader):
    print('================================================')
    print('validation is conducting now!')
    net.eval()
    correct = 0
    total = 0

    label_list = []
    predict_list = []

    for i, (inputs, labels) in enumerate(test_loader):
        # images, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = net(inputs)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        num = (predicted == labels).sum()
        correct = correct + num

        predict_list.extend(predicted)
        label_list.extend(labels.data)

    acc = 100. * correct.item() / total
    fpr, tpr, thresholds = metrics.roc_curve(label_list, predict_list)
    acc_sore = metrics.accuracy_score(label_list, predict_list)
    c_matrix = confusion_matrix(label_list, predict_list)
    print(c_matrix)
    print("ACC is " + str(acc_sore))
    print("AUC is " + str(metrics.auc(fpr, tpr)))
    return acc


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    path = '/data/qiuzhen/cataract_classifi'
    net = torch.load('./model/resnet_acc97.57853403141361.pkl')
    # transform_test = transforms.Compose([
    #     transforms.Resize((576, 192)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    transform_test = transforms.Compose([
        transforms.Resize((540, 180)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = AsoctDataset(root=path, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    acc = val_model(net, test_loader)
