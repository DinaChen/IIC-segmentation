
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

def calculate_accuracy(phi,gt):
    flag=0
    for i in range(0,200):
        for j in range(0,200):
            m=find_max(phi[i][j])
            n=gt[i][j]
            if m==n:
                flag+=1
    accuracy=flag/40000
    print(flag)
    return accuracy
    

def find_max(vec):
    maximum=max(vec[0],vec[1],vec[2])
    return maximum
            
    
def main():
    #phi is 200*200*3,which we get from taining results
    #the following phi is just for test, it's not the true phi
    phi= scipy.io.loadmat('imgs/2.mat')['img']
    gt = scipy.io.loadmat('gt/2.mat')['gt']
    accuracy=calculate_accuracy(phi, gt)
    #print(accuracy)
    
#main()