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
from trainmodel import train as Train
from testmodel import test as Test
import time

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


#
# featurelist = []
# labellist = []
# while(idx<filelen):
#      ithlinestring = rdline(self.filepath , idx)
#      ithlinelist = (ithlinestring.split(','))
#      ithline = np.array(ithlinelist)
#      ithline = ithline.astype('float')
#      featuresrownumpy = ithline[:-1]
#      featuresnumpy = featuresrownumpy.reshape(-1, self.featuresize)
#      features = torch.from_numpy(featuresnumpy.astype(float))
#      features = features.double()
#      labelnumpy = ithline[-1:]
#      label = torch.from_numpy(labelnumpy.astype(float))
#      label = label.double()
#      featurelist.append(features)
#      labellist.append(label)
#      idx = idx+1
#      print(idx)
# features = np.asarray(featurelist)
# labels = np.asarray(labellist)


class FramesDataSet(Dataset):
    def __init__(self, filepath, featuresize=2048):
        self.filepath = filepath
        self.length = getlength(self.filepath)
        self.featuresize = featuresize
        self.c =0
        filelen = getlength(filepath)
        # print(filelen)
        idx = 0
        featurelist = []
        labellist = []
        while(idx<filelen):
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
             featurelist.append(features)
             labellist.append(label)
             idx = idx+1
             print(idx)
        self.features = np.asarray(featurelist)
        self.labels = np.asarray(labellist)
        # print(features.shape)



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # print("here")
        feature = self.features[idx]
        label = self.labels[idx]
        return feature , label
        #  self.c = self.c +1
        # #  print(self.c)
        # #  start_time = time.time()
        #  ithlinestring = rdline(self.filepath , idx)
        #  ithlinelist = (ithlinestring.split(','))
        #  ithline = np.array(ithlinelist)
        #  ithline = ithline.astype('float')
        # #  print("--------")
        # #  end_time = time.time()
        # #  print(end_time - start_time)
        #
        #  featuresrownumpy = ithline[:-1]
        #  featuresnumpy = featuresrownumpy.reshape(-1, self.featuresize)
        #  features = torch.from_numpy(featuresnumpy.astype(float))
        #  features = features.double()
        #  labelnumpy = ithline[-1:]
        #  label = torch.from_numpy(labelnumpy.astype(float))
        #  label = label.double()
        #
        #  return features , label


input_size = 2048

# filename = './sample_output.csv'
# trainfilename = './sample_output.csv'
# testfilename = './sample_output.csv'
trainfilename = '/tmp/anjan/output.csv'
testfilename = '/tmp/anjan/output.csv'
featuresize = 2048

# Model1 smaller one, for proof of concept
seq_len = 10
hidden_size = 20
num_layers = 2
num_epochs = 2
learning_rate = 0.01
batch_size = 5

# Model2
seq_len = 10
hidden_size = 45
num_layers = 2
num_epochs = 2
learning_rate = 0.01
batch_size = 5

# Model3
seq_len = 10
hidden_size = 90
num_layers = 2
num_epochs = 2
learning_rate = 0.01
batch_size = 5


# Model4
seq_len = 10
hidden_size = 20
num_layers = 3
num_epochs = 2
learning_rate = 0.01
batch_size = 5

# Model5
seq_len = 10
hidden_size = 45
num_layers = 3
num_epochs = 2
learning_rate = 0.01
batch_size = 5


# Model6
seq_len = 10
hidden_size = 90
num_layers = 3
num_epochs = 2
learning_rate = 0.01
batch_size = 5

trainloaderT = FramesDataSet(filepath = trainfilename , featuresize = featuresize)
trainloader = DataLoader(trainloaderT, batch_size= batch_size,shuffle=True)

testloaderT = FramesDataSet(filepath = testfilename , featuresize = featuresize)
testloader = DataLoader(testloaderT, batch_size= batch_size,shuffle=True)
#
# from numpy import genfromtxt
# trainnumpy = genfromtxt(trainfilename, delimiter=',')


class VanilaLSTM(nn.Module):
    def __init__(self , input_size , hidden_size , num_layers , seq_len):
        super(VanilaLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size , num_layers =self.num_layers , batch_first  = True)

        # View is applied here
        self.fc = nn.Sequential(
        nn.Linear(self.seq_len * self.hidden_size , self.hidden_size),
        nn.Tanh(),
        nn.Linear(self.hidden_size , 32),
        nn.Hardtanh(),
        nn.Linear(32 , 3) ,
        nn.Softmax())


    def forward(self, x):
        self.lstm.flatten_parameters()
        out , _ = self.lstm(x)
        out = out.contiguous().view(-1, self.seq_len * self.hidden_size )
        out = self.fc(out)

        # out = out.view(-1 ,self.seq_len * self.hidden_size)
        return out



# torch.cuda.set_device(1)
test = VanilaLSTM(input_size=input_size, hidden_size= hidden_size , num_layers=num_layers , seq_len=seq_len)
test.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(test.parameters(), lr=learning_rate)

dtypeim = torch.FloatTensor
dtypelab = torch.LongTensor

params = {}
params['model'] = test
params['num_epochs'] = num_epochs
params['trainloader'] = trainloader
params['optimizer'] = optimizer
params['criterion']= criterion
params['dtypeim'] = dtypeim
params['dtypelab'] = dtypelab
params['batch_size'] = batch_size
params['testloader'] = testloader
params['learning_rate'] = learning_rate




num_epochs = 14
params['num_epochs'] = num_epochs
# params['model'] = test

#Train function
params['model']  = Train(params)

# params['model'] = test
# torch.save(test, './sample.pt')
t = Test(params)
