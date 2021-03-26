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
from PotsdamData import flip,colorJitter,randomCrop



potsdam_preprocessed = 'val2017/bears/croppedBears/'
batch_size = 5
displacements = ['N','S','E','W','NE','NW', 'SE','SW']


# matrix Pt as in equation5, avg over all pixels, images, and transformations
def getPt():

    return 0


# input: a batch of origin images,
#        a batch of transformed image
#        displacement type(string)
# output: 3*3 (potsdam few) or 6*6(potsdam full) matrix, avg over all images in batch.
def getImageAvgMatrix(originBatch, transformedBatch, displacement):

    amountImage = originBatch.shape[0]  # mostly equals to batchsize but the last one

    matrix = torch.zeros(3, 3)

    for id in range(amountImage):
        image_x = originBatch[id]
        image_gx = transformedBatch[id]

        imageMatrix = getPixelMatrixP(image_x, image_gx, displacement)
        matrix = torch.add(matrix, imageMatrix)

    imageAvgMatrix = torch.div(matrix, amountImage)

    return imageAvgMatrix



# input: distribution of images: Φ（x) , Φ（gx）,shape ([200,200,3]) or ([200,200,6])
# output: the matrix p (3*3) or (6,6), averaged over pixels.
def getPixelMatrixP(origin, transformed,displacement):

    ##TODO
    #for flip,  convert the coordinate, maybe create a function for it
    #for random crop, ??
    #for color, nothing

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


def printBatchInfo(iter):
    for bn, batch in enumerate(iter):
        print('Batch No.: '+ str(bn))
        print(batch.shape)


def getDataIterators():

    # prepare Original Dataset
    potsdam_origin = Potsdam(root=potsdam_preprocessed, transforms=None)
    potsdam_origin_loader = torch.utils.data.DataLoader(potsdam_origin, batch_size=batch_size, shuffle=False)
    origin_iter = iter(potsdam_origin_loader)

    # Prepare Flipped Dataset
    potsdam_flip = Potsdam(root=potsdam_preprocessed, transforms=flip)
    potsdam_flip_loader = torch.utils.data.DataLoader(potsdam_flip, batch_size=batch_size, shuffle=False)
    flip_iter = iter(potsdam_flip_loader)

    # Prepare ColorJittered Dataset
    potsdam_color = Potsdam(root = potsdam_preprocessed, transforms=colorJitter)
    potsdam_color_loader = torch.utils.data.DataLoader(potsdam_color, batch_size=batch_size, shuffle=False)
    color_iter = iter(potsdam_color_loader)

    # Prepare RandomCrop Dataset
    potsdam_randomCrop = Potsdam(root=potsdam_preprocessed, transforms=randomCrop)
    potsdam_randomCrop_loader = torch.utils.data.DataLoader(potsdam_randomCrop, batch_size=batch_size, shuffle=False)
    crop_iter = iter(potsdam_randomCrop_loader)

    return origin_iter, flip_iter, color_iter, crop_iter


def main():

    origin_iter, flip_iter,color_iter,crop_iter = getDataIterators()
    printBatchInfo(origin_iter)




    # Assume, output of the model has shape:  [4 * batchSize, 200,200,3]
    # divide them back to the four group: original, flipped, colorChanged, Cropped
    # dummies ~
    output_origin = torch.zeros(batch_size,200,200,3)
    output_colorJitter = torch.zeros(batch_size,200,200,3)
    output_flip = torch.zeros(batch_size,200,200,3)
    output_randomCrop = torch.zeros(batch_size,200,200,3)
















main()