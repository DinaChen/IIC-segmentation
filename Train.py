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
from PotsdamData import Potsdam

potsdam_preprocessed = 'val2017/bears/croppedBears/'
batch_size = 5
displacements = ['N','S','E','W','NE','NW', 'SE','SW']



# input: distribution of Φ（x），Φ（gx）
# output: the matrix p
def getImgMatrixP(origin, transformed,displacement):

    ##TODO
    #for flip,  convert the coordinate, maybe create a function for it
    #for random crop, ??
    #for color, nothing

    # Φ（x），shape ([200,200,3]) or ([200,200,6])
    # multipyle Φ（x），Φgu（gx）has shape ([200，200，3,3]) or ([200,200,6,6])
    # create a new tensor,fill in pixel by pixel.


    matrixPixels = torch.zeros(200,200,3,3)
    matrixP = avgThePixels(matrixPixels)

    return torch.zeros(3,3)
    #return matrixP


# input: matrixP shape[(200,200,3,3)] or [(200,200,6,6)]
# average over all pixels
# output: matrix shape [(3,3)]
def avgThePixels(matrixPixels):



    matrix = torch.zeros(3,3)
    return

# look for what is correct displacement
def getDisplacement(pixel, displacement, movedPixel):

    # consider when displacement is not possible, out of range 0-200

    return (0,0)


def main():
    # prepare Original Dataset
    potsdam_origin = Potsdam(root=potsdam_preprocessed, transforms=None)
    potsdam_origin_loader = torch.utils.data.DataLoader(potsdam_origin, batch_size=batch_size, shuffle=False)
    potsdam_origin_iter = iter(potsdam_origin_loader)

    #for bn, batch in enumerate(potsdam_origin_iter):
    #    print('Batch No.: '+ str(bn))
    #    print(batch.shape)




main()