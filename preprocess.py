import numpy as np
import cv2
import torch
from torchvision import transforms,datasets
import torchvision.transforms as tvt
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

# image is an numpy.ndarray 640 586 3
#Tensor Image is a tensor with (C, H, W) shape, where C is a number of channels, H and W are image height and width.

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
        print(imageId)
        path = from_data_dir + imageId
        image = cv2.imread(path, cv2.IMREAD_COLOR)#.astype(np.uint8)
        scaledImage = scale_by(image, scalar)
        saveAs = to_data_dir + str(imageId)
        cv2.imwrite(saveAs, scaledImage)

# crop every image in from_data_dir and save in folder to_data_dir
def croppImagesFile():

    from_data_dir = 'val2017/bears/scaledBears/'
    to_data_dir = 'val2017/bears/croppedBears/'

    centerCrop = tvt.transforms.CenterCrop(128)

    for imageId in os.listdir(from_data_dir):
        path = from_data_dir + imageId
        image = Image.open(path)
        croppedImage = centerCrop(image)
        saveAs = to_data_dir + str(imageId)
        croppedImage.save(saveAs)

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

def main():
    colorJitterImageFile()
    #im = Image.open("val2017/bears/croppedBears/000000073118.jpg")
    #i = ImageOps.mirror(Image.open('val2017/bears/croppedBears/000000073118.jpg'))
    #i.show()
    #scaleImagesFile()
    #cropImagesFile()
    #flipImageFile()
    #colorJitter = tvt.transforms.ColorJitter(jitter_brightness, jitter_contrast, jitter_saturation, jitter_hue)
    #jitImage = colorJitter(im)
    #im.show()
    #jitImage.show()

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






