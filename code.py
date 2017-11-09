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


def rdline(filename, i):
    number = i
    with open(filename) as fp:
        for line in fp:
            if number == 0:
                return line
            number-=1

def getlength(filename):
    number = 0
    with open(filename) as fp:
        for line in fp:
            number+=1
    return number


class FramesDataSet(Dataset):
    def __init__(self, filepath, featuresize=2048):
        self.filepath = filepath
        self.length = getlength(self.filepath)
        self.featuresize = featuresize

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #  print(idx)
         ithlinestring = rdline(self.filepath , idx)
         ithlinelist = (ithlinestring.split(','))
         ithline = np.array(ithlinelist)
         ithline = ithline.astype('float')

         featuresrownumpy = ithline[:-1]
         featuresnumpy = featuresrownumpy.reshape(-1, self.featuresize)
         features = torch.from_numpy(featuresnumpy.astype(float))
         features = features.double()

         labelnumpy = ithline[-1:]
         label = torch.from_numpy(labelnumpy.astype(float))
         label = label.double()

         return features , label


batch_size = 5
filename = './sample_output.csv'
featuresize = 2048

trainloaderT = FramesDataSet(filepath = filename , featuresize = featuresize)
trainloader = DataLoader(trainloaderT, batch_size= batch_size,shuffle=True)

testloaderT = FramesDataSet(filepath = filename , featuresize = featuresize)
testloader = DataLoader(testloaderT, batch_size= batch_size,shuffle=True)
