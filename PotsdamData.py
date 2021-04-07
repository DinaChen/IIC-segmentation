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
import scipy.io




class Potsdam(VisionDataset):
    def __init__(self, root):
        #super()._init_(root, transforms)
        self.root = root
        self.images = os.listdir(root)
        self.groundTruth = os.listdir(root)

    def __getitem__(self, index):

        path = self.root + self.images[index]
        image = scipy.io.loadmat(path)['img'].astype(np.uint8)

        # get two types of transformed images
        origin = Image.fromarray((image))
        flipImg = Potsdam.flip(image)
        jitter = Potsdam.jitter(image)

        # PIL to float tensor
        originTensor = torchvision.transforms.functional.pil_to_tensor(origin).float()
        flipTensor = torchvision.transforms.functional.pil_to_tensor(flipImg).float()
        jitterTensor = torchvision.transforms.functional.pil_to_tensor(jitter).float()

        tensor = torch.stack((originTensor, flipTensor, jitterTensor), dim=0)


        return  tensor


    def __len__(self):
        return len(self.images)


    # The 2 transformation functions

    # input: npArray
    def flip(image):
        image = Image.fromarray((image))
        image = ImageOps.mirror(image)
        # image type: PIL

        return image

    # input: an nparray image
    def jitter(image):

        ## ?????????????????????????
        hue = (-0.5, 0.5)
        contrast = (0, 1)
        saturation = (0, 1)
        brightness = (0, 2)

        collorJitter = tvt.transforms.ColorJitter(brightness, contrast, saturation, hue)

        # Seperate rgb and ir channels, NpArray
        img_ir = image[:, :, 3]
        img_ir = img_ir.astype(np.float32) / 255.
        img_rgb = image[:, :, :3]

        # npArray to PIL for jitter function
        imgRGB = Image.fromarray(img_rgb.astype(np.uint8))

        # Jittered image is in PIL format
        imgJit = collorJitter(imgRGB)
        # imgJit.show()

        # PIL to NpArray for concatenation
        imgJit = np.array(imgJit)
        imgJit = imgJit.astype(np.float32) / 255.

        # Concatenate IR back on before spatial warps
        # may be concatenating onto just greyscale image
        # grey/RGB underneath IR
        imgJit = np.concatenate([imgJit, np.expand_dims(img_ir, axis=2)], axis=2)

        # NpArray to PIL
        imgJit = Image.fromarray((imgJit * 255).astype(np.uint8))
        #imgJit.show()

        return imgJit
















