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


def train(model, loss, optimizer, x_val, y_val, metric):
    model.train()
    x = Variable(x_val.cuda())
    y = Variable(y_val.cuda())

    optimizer.zero_grad()
    output = model.forward(x)
    avg_output = (output[0]+output[1]+output[2]+output[3]+output[4]+output[5])/6
    hloss = loss.forward(output[0], y)
    for i in range(5):
        hloss += loss.forward(output[i+1], y)
    hloss.backward()
    optimizer.step()
    # print(output.size())
    # batchsize, _, _, _ = x.size()
    _, predicted = torch.max(avg_output, 1)
    metric.update_metrics_batch(predicted, y)
    correct = (predicted == y).sum()
    # distance_loss = []
    # for i in range(len(x_val)):
    #     distance_loss += [abs(predicted[i]-y[i])]

    return hloss.data[0], correct
