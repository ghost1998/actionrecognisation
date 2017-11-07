# actionrecognisation
Feature extraction is done. Check the code testgetfeatures to understand how to get features for each image.

It is basicly the below code.

a.extract(image) returns numpyfeatures of an image of any size. It takes care of resizing with cubic interpolation. 

I have to make the lstm once the dataset is done.

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
from getfeatures import getfeatures

img = cv2.imread('frozen.jpg')
a = getfeatures()
print(a.extract(img.astype(float)))

