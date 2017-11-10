# Todo
# 1)Get featuresfrom inception architechture which takes input as an image
# 2)Then use the above code to convert the input video frames to features
# 3)Then feed the featues to LSTM to predict



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
# my_data = genfromtxt('output.csv', delimiter=',')
# size
# img = cv2.imread('frozen.jpg')
# res = cv2.resize(img,(299, 299), interpolation = cv2.INTER_CUBIC)
# ale = cv2.resize(img,(224, 224), interpolation = cv2.INTER_CUBIC)
# cv2.imshow('image',res)
# cv2.waitKey(0)


transform = transforms.ToTensor()
"""
model = models.alexnet(pretrained=True)
value = torch.from_numpy(ale)

# Convert to torch tensor
value = transform(ale.astype(float))

# Convert to variable for autograd
test_value = Variable(value)
test_value = test_value.float()
test_value = test_value.unsqueeze(0)
prediction = model(test_value)

new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier
"""

# Make the data
# value = transform(res.astype(float))
#
# # Convert to variable for autograd
# test_value = Variable(value)
# test_value = test_value.float()
#
# #Make it 4D
# test_value = test_value.unsqueeze(0)
#
# # Define the pretrained model
# inception = models.inception_v3(pretrained=True)
# # Remove the last layer
# new_classifier = nn.Sequential(*list(inception.children())[:-1])
# inception.classifier = new_classifier
# prediction = inception(test_value)



class inceptionfeatures(nn.Module):
    def __init__(self, inception, transform_input=False):
        super(inceptionfeatures, self).__init__()

        self.aux_logits = inception.aux_logits
        self.transform_input = inception.transform_input
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e

        if inception.aux_logits:
            self.AuxLogits = inception.AuxLogits
        self.Mixed_7a =  inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        self.fc = inception.fc

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        # x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x

# in1 = inceptionfeatures(inception)
# prediction = in1(test_value)



#
# class MyInceptionFeatureExtractor(nn.Module):
#     def __init__(self, inception, transform_input=False):
#         super(MyInceptionFeatureExtractor, self).__init__()
#         self.transform_input = transform_input
#         self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
#         self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
#         self.Conv2d_2b_3x3 = inception.Conv2d_3a_3x3
#         self.Conv2d_3b_1x1 = inception.Conv2d_3b_3x3
#         self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
#         self.Mixed_5b = inception.Mixed_5b
#         # stop where you want, copy paste from the model def
#
#     def forward(self, x):
#         if self.transform_input:
#             x = x.clone()
#             x[0] = x[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
#             x[1] = x[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
#             x[2] = x[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
#         # 299 x 299 x 3
#         x = self.Conv2d_1a_3x3(x)
#         # 149 x 149 x 32
#         x = self.Conv2d_2a_3x3(x)
#         # 147 x 147 x 32
#         x = self.Conv2d_2b_3x3(x)
#         # 147 x 147 x 64
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
#         # 73 x 73 x 64
#         x = self.Conv2d_3b_1x1(x)
#         # 73 x 73 x 80
#         x = self.Conv2d_4a_3x3(x)
#         # 71 x 71 x 192
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
#         # 35 x 35 x 192
#         x = self.Mixed_5b(x)
#         # copy paste from model definition, just stopping where you want
#         return x

# inception = torchvision.models['inception_v3_google']
# my_inception = MyInceptionFeatureExtractor(inception)


inception = models.inception_v3(pretrained=True)
inceptionfeaturesmodel = inceptionfeatures(inception)
prediction = inceptionfeaturesmodel(test_value)

def rdline(filename, i):
    number = i-1
    with open(filename) as fp:
        for line in fp:
            number-=1
            if number == 0:
                return line

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

from numpy import genfromtxt
numpydata = genfromtxt('sample_output.csv', delimiter=',')
featuresize = 2048


# ith Sample
numpydata[i].shape
length = len(numpydata)


ithline = numpydata[i]
featuresrownumpy = ithline[:-1]
featuresnumpy = featuresrownumpy.reshape(-1, featuresize)
features = torch.from_numpy(featuresnumpy.astype(float))
# features = features.double()

labelnumpy = ithline[-1:]
label = torch.from_numpy(labelnumpy.astype(float))
# label = label.double()

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
testloaderT = FramesDataSet(filepath = './sample_output.csv' , featuresize = 2048)
testloader = DataLoader(testloaderT, batch_size= batch_size,shuffle=True)

len(testloader)

for num,i in enumerate(testloader):
    print(num)
    print(i)
    # print(type((len(i[0]))))
    # print(len(i[1]))
    print("--------")



batch_size = 7
seq_len = 10
num_layers =3
hidden_size = 4
input_size = 2048

lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size , num_layers =num_layers , batch_first  = True)
inp = torch.autograd.Variable(torch.randn(batch_size, seq_len, input_size))
out, _ = lstm(inp)
# h_0 = torch.autograd.Variable(torch.randn(batch_size, seq_len, hidden_size))
# c_0 = torch.autograd.Variable(torch.randn(batch_size, seq_len, hidden_size))
# h_0 = torch.autograd.Variable(torch.randn( seq_len,batch_size, hidden_size))
# c_0 = torch.autograd.Variable(torch.randn( seq_len,batch_size, hidden_size))

# hidden = (h_0 , c_0)
# out, hidden = lstm(inp)
