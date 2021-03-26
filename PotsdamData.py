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

class Potsdam(VisionDataset):
    def __init__(self, root, transforms = None):
        #super()._init_(root, transforms)
        self.root = root
        self.images = os.listdir(root)
        self.groundTruth = os.listdir(root)
        self.transforms = transforms

    def __getitem__(self, index):

        path = self.root + self.images[index]
        image = cv2.imread(path,cv2.IMREAD_COLOR).astype(np.uint8)

        image = Image.fromarray((image))#Transform npArray to PIL

        if self.transforms:

            image = self.transforms(image)

        return  torchvision.transforms.functional.pil_to_tensor(image)


    def __len__(self):
        return len(self.images)

# The 3 transformation functions
def flip(image):

    image = ImageOps.mirror(image) #image type: PIL

    return image

def colorJitter(image):

    ## ?????????????????????????
    hue = (-0.5, 0.5)
    contrast = (0, 1)
    saturation = (0, 1)
    brightness = (0, 5)

    collorJitter = tvt.transforms.ColorJitter(brightness, contrast, saturation, hue)
    image = collorJitter(image)

    return image

def randomCrop(image):

    return image

def main():

    # path of (scaled and cropped) data
    potsdam_preprocessed = 'val2017/bears/croppedBears/'
    batch_size = 5

    # prepare Original  Dataset
    potsdam_origin = Potsdam(root = potsdam_preprocessed)
    potsdam_origin_loader = torch.utils.data.DataLoader(potsdam_origin, batch_size=batch_size, shuffle=False)
    potsdam_origin_iter = iter(potsdam_origin_loader)

    # Prepare Flipped Dataset
    potsdam_flip = Potsdam(root = potsdam_preprocessed, transforms=flip)
    potsdam_flip_loader = torch.utils.data.DataLoader(potsdam_flip, batch_size=batch_size, shuffle=False)
    potsdam_flip_iter = iter(potsdam_flip_loader)

    # Prepare ColorJittered Dataset
    potsdam_color = Potsdam(root = potsdam_preprocessed, transforms=colorJitter)
    potsdam_color_loader = torch.utils.data.DataLoader(potsdam_color, batch_size=batch_size, shuffle=False)
    potsdam_color_iter = iter(potsdam_color_loader)

    # Prepare RandomCrop Dataset
    potsdam_randomCrop = Potsdam(root = potsdam_preprocessed, transforms=randomCrop)
    potsdam_randomCrop_loader = torch.utils.data.DataLoader(potsdam_randomCrop, batch_size=batch_size, shuffle=False)
    potsdam_randomCrop_iter = iter(potsdam_randomCrop_loader)



  #  for bn, batch in enumerate(potsdam_flip_iter):
  #      print('Batch No.: '+ str(bn))
  #      print(batch.shape)

       #for img in batch:
       #     cv2.imshow(str(bn), img.numpy())
       #     cv2.waitKey(0)



main()