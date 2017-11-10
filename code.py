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




trainloaderT = FramesDataSet(filepath = filename , featuresize = featuresize)
trainloader = DataLoader(trainloaderT, batch_size= batch_size,shuffle=True)

testloaderT = FramesDataSet(filepath = filename , featuresize = featuresize)
testloader = DataLoader(testloaderT, batch_size= batch_size,shuffle=True)



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
        nn.Linear(seq_len * hidden_size , hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size , 32),
        nn.Hardtanh(),
        nn.Linear(32 , 12))


    def forward(self, x):
        out , _ = self.lstm(x)
        out = out.contiguous().view(-1, seq_len * hidden_size )
        out = self.fc(out)

        # out = out.view(-1 ,self.seq_len * self.hidden_size)
        return out

input_size = 2048
num_epochs = 10
learning_rate = 0.001
batch_size = 5
filename = './sample_output.csv'
featuresize = 2048
seq_len = 30
hidden_size = 5








# torch.cuda.set_device(1)
test = VanilaLSTM(input_size=input_size, hidden_size= hidden_size , num_layers=num_layers , seq_len=seq_len)
# test.cuda()
# criterion = nn.MSELoss()
# criterion =  nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(test.parameters(), lr=learning_rate)
# dtype = torch.cuda(0).FloatTensor
dtype = torch.FloatTensor



# Training
for epoch in range(num_epochs):
    print("Epoch number ----->" + str(epoch))
    for i, it in enumerate(trainloader):
        # print(it[0].size())
        # print(it[1].size())
        images = it[0]
        labels = it[1]
        # print((images.size()))
        images = Variable(images.type(dtype))
        labels = Variable(labels.type(dtype))
        labels = labels.type(torch.LongTensor)

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
        # print("done")

        if (i+1) % 3 == 0:
            print(len(trainloader))
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, len(trainloader)//batch_size, loss.data[0]))
