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
