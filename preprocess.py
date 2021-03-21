import numpy as np
import cv2
import torch
from torchvision import transforms,datasets
import torchvision.transforms as tvt
from PIL import Image
import os
from pathlib import Path
import shutil

scalar = 1 / 3
######################  Data pre-processing  ###########
# "For both COCO datasets, input images are shrunk
# by two thirds and cropped to 128 × 128 pixels, 128 * 128"

# image is an numpy.ndarray 640 586 3
#Tensor Image is a tensor with (C, H, W) shape, where C is a number of channels, H and W are image height and width.


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
        print(imageId)
        path = from_data_dir + imageId
        image = cv2.imread(path, cv2.IMREAD_COLOR)#.astype(np.uint8)
        scaledImage = scale_by(image, scalar)
        desName = to_data_dir + str(imageId)
        cv2.imwrite(desName, scaledImage)

# crop every image in from_data_dir and save in folder to_data_dir
def croppedImagesFile():

    from_data_dir = 'val2017/bears/scaledBears/'
    to_data_dir = 'val2017/bears/croppedBears/'

    centerCrop = tvt.transforms.CenterCrop(128)

    for imageId in os.listdir(from_data_dir):
        path = from_data_dir + imageId
        image = Image.open(path)
        croppedImage = centerCrop(image)
        desName = to_data_dir + str(imageId)
        croppedImage.save(desName)


def main():

    scaleImagesFile()
    croppedImagesFile()

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






