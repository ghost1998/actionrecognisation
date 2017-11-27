from numpy import genfromtxt
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import torch.utils.data as utils
import cv2
import csv
from torch.utils.data import Dataset, DataLoader
import time

def test(params):
    test = params['model']
    num_epochs = params['num_epochs']
    trainloader = params['trainloader']
    # optimizer = params['optimizer']
    criterion = params['criterion']
    dtypeim = params['dtypeim']
    dtypelab =  params['dtypelab']
    batch_size = params['batch_size']
    testloader = params['testloader']
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = Variable(images.type(dtypeim))
        labels = Variable(labels.type(dtypelab))
        images = images.cuda()
        labels = labels.cuda()
        # st_time = time.time()
        outputs = test(images)
        # print(time.time() - st_time)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        print(labels)
        total += labels.size(0)
        correct += (predicted == labels[:, 0].data).sum()
        print(correct)
        print(total)
        print("------------")

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
