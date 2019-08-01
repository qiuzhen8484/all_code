from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
import math
from torch.nn import init
from metrics import Metrics


def train(model, loss, optimizer, x_val, y_val):
    model.train()
    x = Variable(x_val.cuda())
    y = Variable(y_val.cuda())

    optimizer.zero_grad()
    output = model.forward(x)
    hloss = loss.forward(output[0], y)
    hloss.backward()
    optimizer.step()
    _, predicted = torch.max(output[0], 1)
    correct = (predicted == y).sum()

    # mask1 = output[0][:, 1, :, :]
    # hloss = loss.forward(mask1, y.float())
    # # hloss = loss.forward(output[0], y)
    # hloss.backward()
    # optimizer.step()
    # _, predicted = torch.max(output[0], 1)
    # correct = (predicted == y).sum()

    # avg_output = (output[0] + output[1] + output[2] + output[3] + output[4] + output[5]) / 6
    # hloss = loss.forward(output[0][:, 1, :, :], y.float())
    # for i in range(5):
    #     hloss += loss.forward(output[i+1][:, 1, :, :], y.float())
    # hloss.backward()
    # optimizer.step()
    # _, predicted = torch.max(avg_output, 1)
    # correct = (predicted == y).sum()
    # avg_output = (output[0] + output[1] + output[2] + output[3] + output[4] + output[5]) / 6
    # hloss = loss.forward(output[0], y)
    # for i in range(5):
    #     hloss += loss.forward(output[i + 1], y)
    # hloss.backward()
    # optimizer.step()
    # _, predicted = torch.max(avg_output, 1)
    # correct = (predicted == y).sum()

    return hloss.data[0], correct, predicted.cpu()


def iou(pred, target, n_classes=2):
    # n_classes ：the number of classes in your dataset
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)

def Dice(pred, target, n_classes=2):
    # n_classes ：the number of classes in your dataset
    dice = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore Dice for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0]
        if union == 0:
            dice.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            dice.append(float(2 * intersection) / float(max(union, 1)))
    return np.array(dice)


def IOU(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    # print(int(y_true_f.sum()))
    if int(y_true_f.sum()) == 0 and int(y_pred_f.sum()) == 0:
        return 1.0
    else:
        # print((y_true_f.sum() + y_pred_f.sum() - intersection))
        #return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
        return float(intersection / (y_true_f.sum() + y_pred_f.sum() - intersection))


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1.0

        input_flat = input.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = input_flat * target_flat
        # intera = intersection.long().sum()
        loss = ((2 * intersection.sum() + smooth)).float() / ((input_flat.sum() + target_flat.sum() + smooth)).float()
        # loss1 = (2 * intersection.long().sum() + smooth) / (input_flat.long().sum() + target_flat.long().sum() + smooth)
        loss = 1.0 - loss

        return loss

