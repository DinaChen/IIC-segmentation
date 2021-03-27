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
displacements = ['up','down','left','right','upright','upleft','downright','downleft']
n = 3  #for potsdam-3
#n = 6 #for potsdam-6


#not sure yet what the ouput from model looks
def getLoss():

    # Now we have the output from the model of one batch, we want to calculate the information(loss) of this batch.
    # Assume: output of the model has shape:  [4 * batchSize, 200,200,n]
    # divide them back to the four group: original, flipped, colorChanged, Cropped
    # dummies ~
    output_origin = torch.zeros(batch_size, 200, 200, n)
    output_color = torch.zeros(batch_size, 200, 200, n)
    output_flip = torch.zeros(batch_size, 200, 200, n)
    output_crop = torch.zeros(batch_size, 200, 200, n)

    matrixP = getMatrixP(output_origin, output_flip, output_color, output_crop)
    information = getInformation(matrixP)


# equation 3
def getInformation(matrixP):

    return 0


# correspond to avgPt in equation 5, everaged over all displacements T.
def getMatrixP(output_origin, output_flip, output_color, output_crop):

    matrix = torch.zeros(n,n)
    for displacement in displacements:

        print(displacement)

        transform1 = getImageAvgMatrix(output_origin, output_flip, displacement)
        transform2 = getImageAvgMatrix(output_origin, output_color, displacement)
        transform3 = getImageAvgMatrix(output_origin, output_crop, displacement)

        transformAvgMatrix = (transform1 + transform2 + transform3) / 3;
        matrix = matrix + transformAvgMatrix

    matrix = matrix / len(displacements)
    return matrix


# input: a batch of origin images,
#        a batch of transformed image
#        displacement type(string)
# output: 3*3 (potsdam few) or 6*6(potsdam full) matrix, avg over all images in batch.
def getImageAvgMatrix(originBatch, transformedBatch, displacement):

    amountImage = originBatch.shape[0]  # mostly equals to batchsize but the last one

    matrix = torch.zeros(n, n)

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


    matrixPixels = torch.zeros(200,200,n,n)
    matrixP = avgThePixels(matrixPixels)

    return torch.zeros(n,n)
    #return matrixP


# input: matrixP shape[(200,200,3,3)] or [(200,200,6,6)]
# average over all pixels
# output: matrix shape [(3,3)] or [(6,6)]
def avgThePixels(matrixPixels):



    matrix = torch.zeros(n,n)
    return
############################### Helper Functions #################################

#get corresponding pixel coordinate (h,w) for a horizontally flipped image
def flipCoordinate(pixel):

    h,w = pixel
    return (h, 199-w)

# get corresponding pixel coordinate (h,w) for the transformed image, given the displacmenet type.
# 'u+t' in equation 5 in IIC paper
def getDisplacement(pixel, displacement):

    h,w = pixel
    move_h, move_w = getDirection(displacement)
    new_h = h + move_h
    new_w = w + move_w

    # when displacement is not possible: out of range 0-200, then not move.
    if(new_h < 0 or new_h>199):
        new_h = h
    if (new_w < 0 or new_w > 199):
        new_w = w

    return (new_h, new_w)

def getDirection(x):
    return {
        'up': (-1,0),
        'down': (1,0),
        'left':(0,-1),
        'right':(0,1),
        'upright':(-1,1),
        'upleft':(-1,-1),
        'downright':(1,1),
        'downleft':(1,-1),
    }[x]

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

   print(flipCoordinate((199, 199)))


    #origin_iter, flip_iter,color_iter,crop_iter = getDataIterators()
    #printBatchInfo(origin_iter)

    # Train the model batch by batch
    #if(origin_iter.hasNext()):   ##right？

        ## Data for one Batch
     #   batch_origin = next(origin_iter)
     #   batch_flip = next(flip_iter)
     #   batch_color = next(color_iter)
     #   batch_crop = next(crop_iter)

        ## concat them and feed into Model

    #when get output
    #getLoss()






main()