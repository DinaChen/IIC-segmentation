from pathlib import Path
import shutil
import os
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as tvt
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.functional as F
from PIL import Image
from PIL import ImageOps

import torch.nn as nn
import torch.nn.functional as F

import scipy.io



# calculate the accuracy for whole test set
# imgs: [testSetSize, c, 200, 200]
# gt: [testSetSize, 200, 200]
def calculate_accuracy_all(imgs, gts, n):
    assert imgs.shape[0] == gts.shape[0]
    testSetSize = imgs.shape[0]

    accuracy = 0
    for i in range(testSetSize):

        accu = calculate_accuracy(imgs[i], gts[i],n)
        accuracy = accuracy + accu

        #print(accu)

    avgAccuracy = accuracy / testSetSize
    return avgAccuracy



# calculate the accuracy of one image
# imgs: [c, 200, 200]
# gt: [200,200]
def calculate_accuracy(phi,gt,n):

    # first convert imgs shape to [200, 200, c]
    phi = phi.permute(1,2,0)

    flag=0
    for i in range(0,200):
        for j in range(0,200):

            # find index with maximum probability
            img_class = torch.argmax(phi[i][j])
            gt_class = gt[i][j]

            if(sameClass(img_class,gt_class,n)):
                flag = flag+1

    accuracy=flag/40000

    return accuracy
    

def sameClass(imgClass, gtClass, n):

    if n == 6:
        return imgClass==gtClass

    # if n = 3
    if (gtClass == 5): gtClass = 1
    if (gtClass == 4): gtClass = 0
    if (gtClass == 3): gtClass = 2

    return imgClass==gtClass
    
