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
from inceptionfeatures import inceptionfeatures

class getfeatures():
    def __init__(self):
        inception = models.inception_v3(pretrained=True)
        self.inceptionfeaturesmodel = inceptionfeatures(inception)


    def extract(self, img):
        reshapedimage = cv2.resize(img,(299, 299), interpolation = cv2.INTER_CUBIC)
        transform = transforms.ToTensor()
        transformedimage = transform(reshapedimage.astype(float))
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        # normalizedimage = normalize(reshapedimage.astype(float))
        image_variable = Variable(transformedimage)
        image_variable = image_variable.float()
        image_variable = image_variable.unsqueeze(0)
        prediction = self.inceptionfeaturesmodel(image_variable)
        return prediction[0]
