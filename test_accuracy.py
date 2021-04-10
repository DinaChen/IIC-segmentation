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


def calculate_average_accuracy(gt_path,whole_phi,n):
    
    path_list=sort_path(gt_path) #sort gt path
    num=0
    for i in range(0,n):
        gt = scipy.io.loadmat(gt_path+path_list[i])['gt']
        num+=calculate_accuracy(whole_phi[i], gt)
    num/=n
    return num

def calculate_accuracy(phi,gt):
    flag=0
    for i in range(0,200):
        for j in range(0,200):
            m=find_max(phi[i][j])
            n=gt[i][j]
            if m==n:
                flag+=1
    accuracy=flag/40000
    # add "%"
    accuracy*=100
    return accuracy
    

def find_max(vec):
    maximum=max(vec[0],vec[1],vec[2])
    return maximum

def sort_path(path):
    
 
    path_list = os.listdir(path)
 
 
    path_list.sort(key=lambda x:int(x.split('.')[0]))
 
    #print(path_list)
    return path_list

    
           
    
def main():
    #phi is n*200*200*c, which we get from taining results
    #the following phi is just for test, it's not the true phi
    phi=torch.zeros(5400,200, 200, 3)
    n=5400
    #gt = scipy.io.loadmat('gt/3.mat')['gt']
    #gt=torch.ones(200, 200) 
    gt_path = 'gt/'
    accuracy=calculate_average_accuracy(gt_path, phi, n)
    print(accuracy,"%")
    
main()
