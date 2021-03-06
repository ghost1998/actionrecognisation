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

def train(params):
    test = params['model']
    num_epochs = params['num_epochs']
    trainloader = params['trainloader']
    optimizer = params['optimizer']
    criterion = params['criterion']
    dtypeim = params['dtypeim']
    dtypelab =  params['dtypelab']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    # optimizer = torch.optim.Adam(test.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        print("Epoch number ----->" + str(epoch))
        for i, it in enumerate(trainloader):
            st_time = time.time()
            # print(it[0].size())
            # print(it[1].size())
            images = it[0]
            labels = it[1]
            # print((images.size()))
            images = Variable(images.type(dtypeim) , requires_grad=False)
            labels = Variable(labels.type(dtypelab), requires_grad=False)
            images = images.cuda()
            labels = labels.cuda()
            # labels = labels.type(torch.LongTensor)

            # images = Variable(images.type(dtype)).cuda()
            # labels = Variable(labels.type(dtype)).cuda()
            # print("conv. to varialbes and cast to dtype")
            # break
            # images = Variable(images)
            # labels = Variable(labels)

            optimizer.zero_grad()
            # print(images)
            # print(labels)
            # break
            output = test(images)
            # print(output)
            # print("got output")
            # break
            # print("got output")
            # print(output.size())
            # break
            loss = criterion(output, labels[:,0])
            # print("got loss")
            loss.backward()
            # print("back")
            optimizer.step()
            print(time.time() - st_time)

            # In here if loss.data[0] is least, save the model. google how to save a model. I ma not sure
            # print("done")
            if (i+1) % 3 == 0:
                # print(len(trainloader))
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, len(trainloader)//batch_size, loss.data[0]))
    return test
