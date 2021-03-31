import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms,datasets
import torchvision.transforms as tvt
from torch.autograd import Variable
from PIL import Image
from PIL import ImageOps
import os
from pathlib import Path
import shutil


scalar = 1 / 3
jitter_hue = 0.125
jitter_contrast = 0.4
jitter_saturation = 0.4
jitter_brightness = 0.4

######################  Data pre-processing  ###########
# "For both COCO datasets, input images are shrunk
# by two thirds and cropped to 128 × 128 pixels, 128 * 128"


###################### Transformation G  ##############
# horizontal flip
# random crop
# random color changes in hue, saturation and brightness


# to scale the image by scalar
def scale_by(image, scalar):

    width = int(image.shape[1] * scalar)
    height = int(image.shape[0] * scalar)
    newSize = (width, height)

    return cv2.resize(image, newSize)


# scale every image in from_data_dir and save in folder to_data_dir
def scaleImagesFile():

    from_data_dir = 'val2017/bears/bears/'
    to_data_dir = 'val2017/bears/scaledBears/'

    for imageId in os.listdir(from_data_dir):
        #print(imageId)
        path = from_data_dir + imageId
        image = cv2.imread(path, cv2.IMREAD_COLOR)#.astype(np.uint8)
        scaledImage = scale_by(image, scalar)
        saveAs = to_data_dir + str(imageId)
        cv2.imwrite(saveAs, scaledImage)

# crop every image in from_data_dir and save in folder to_data_dir
def croppImagesFile():

    from_data_dir = 'val2017/bears/scaledBears/'
    to_data_dir = 'val2017/bears/croppedBears/'

    centerCrop = tvt.transforms.CenterCrop(200)

    for imageId in os.listdir(from_data_dir):
        path = from_data_dir + imageId
        image = Image.open(path)
        croppedImage = centerCrop(image)
        saveAs = to_data_dir + str(imageId)
        croppedImage.save(saveAs)


# implemented in PotsdamData.py, ignore here
# flip every image in from_data_dir and save in folder to_data_dir
def flipImageFile():

    from_data_dir = 'val2017/bears/croppedBears/'
    to_data_dir = 'val2017/bears/flippedBears/'

    for imageId in os.listdir(from_data_dir):
        path = from_data_dir + imageId
        flippedImage = ImageOps.mirror(Image.open(path))
        saveAs = to_data_dir + str(imageId)
        flippedImage.save(saveAs)


# perform color transformation for every image in from_data_dir and save in folder to_data_dir
# jitter: brightness, contrast, saturation, hue, parameters same as in original IIC code
def colorJitterImageFile():

    from_data_dir = 'val2017/bears/croppedBears/'
    to_data_dir = 'val2017/bears/colorJitBears/'

    colorJitter = tvt.transforms.ColorJitter(jitter_brightness, jitter_contrast, jitter_saturation, jitter_hue)

    for imageId in os.listdir(from_data_dir):
        path = from_data_dir + imageId
        image = Image.open(path)
        jitteredImage = colorJitter(image)
        saveAs = to_data_dir + str(imageId)
        jitteredImage.save(saveAs)

# probably wont be used anymore
def sobel():

    image = cv2.imread('val2017/bears/bears/000000044068.jpg', cv2.IMREAD_COLOR)  # .astype(np.uint8)
    print(image.shape)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    absx = np.absolute(sobel_x)
    absy = np.absolute(sobel_y)

    sobelx = np.uint8(absx)
    sobely = np.uint8(absy)

    # blend them
    #sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    print(sobelx)

    #cat = torch.cat([torch.from_numpy(sobelx), torch.from_numpy(sobely)], dim=1)
    #print(cat.shape)
    cv2.imshow('x', sobelx)

    # cv2.imshow('sobelx', sobelx)
    # cv2.imshow('sobely', sobely)
    # cv2.imshow('blend', grad)

    cv2.waitKey(0)
    # sol = sobel_process( torch.from_numpy(image),True,False)
    # print(sol.shape)

def main():
  #croppImagesFile()

  # 33 48  11:16
  #
 main()


    #pre_transform = transforms.Compose([#transforms.Resize(255),
    #                                transforms.CenterCrop(128),
    #                                transforms.ToTensor()])

    # t = torch.nn.Sequential(
    #    transforms.TenCrop((128,128))
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #)


    #cropImage = transforms.TenCrop((128,128))
    #print(cropImage.shape)

#    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)






