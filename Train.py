import os
import numpy as np
import cv2
import torch
torch.cuda.is_available()
import torchvision
import torchvision.transforms as tvt
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.functional as F
from PIL import Image
from PIL import ImageOps
from PotsdamData import Potsdam
from PotsdamData import flip,colorJitter,randomCrop
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vgg import VGGTrunk, VGGNet



potsdam_preprocessed = 'val2017/bears/croppedBears/'
batch_size = 5
displacements = ['up','down']#,'left','right'] #'upright','upleft','downright','downleft']
#n = 3  #for potsdam-3
n = 6 #for potsdam-6


# equation 3
def getInformation(matrixP):

    # symmetrize matrixP
    transpose = torch.transpose(matrixP, 0,1)
    matrix = (matrixP + transpose)/2

    information = 0

    for h in range(matrix.shape[0]):
        p_h = torch.sum(matrix[h])

        for w in range (matrix.shape[1]):
            p_w = torch.sum(matrix[w])

            p_hw = matrix[h][w]

            info = p_hw * torch.log(p_hw/p_w*p_h)
            information = info+information

    return information


# temperal! for trying to use the model

def getLossT(output_origin, output_flip):

    totalLoss = 0

    for displacement in displacements:

        #There are 8 displacements
        print(displacement)

        transform1 = getImageAvgMatrix(output_origin, output_flip, displacement, 'flip')

        transformAvgMatrix = (transform1 + transform1 + transform1) / 3;
        loss = getInformation(transformAvgMatrix)
        totalLoss = totalLoss + loss

    avgLoss = totalLoss / 8

    return avgLoss

# Equation 5, objective function
# for each displacement, calculate the matrix, which is averaged over 3 transformations
# calculate the loss, then average over the displacements.
def getLoss(output_origin, output_flip, output_color, output_crop):

    totalLoss = 0

    for displacement in displacements:

        #There are 8 displacements
        print(displacement)

        transform1 = getImageAvgMatrix(output_origin, output_flip, displacement, 'flip')
        transform2 = getImageAvgMatrix(output_origin, output_color, displacement, 'color')
        transform3 = getImageAvgMatrix(output_origin, output_crop, displacement, 'crop')

        transformAvgMatrix = (transform1 + transform2 + transform3) / 3;
        loss = getInformation(transformAvgMatrix)
        totalLoss = totalLoss + loss

    avgLoss = totalLoss / 8

    return avgLoss


# input: a batch of origin images,
#        a batch of transformed image
#        displacement type(string)
#        type of transformation
# output: 3*3 (potsdam few) or 6*6(potsdam full) matrix, avg over all images in batch.
def getImageAvgMatrix(originBatch, transformedBatch, displacement,transformType):

    amountImage = originBatch.shape[0]  # mostly equals to batchsize but the last one

    matrix = torch.zeros(n, n)

    for id in range(amountImage):
        print(id)

        image_x = originBatch[id]
        image_gx = transformedBatch[id]

        imageMatrix = getPixelMatrixP(image_x, image_gx, displacement, transformType)
        matrix = torch.add(matrix, imageMatrix)

    imageAvgMatrix = torch.div(matrix, amountImage)

    return imageAvgMatrix



# input: distribution of images: Φ（x) , Φ（gx）, shape ([200,200,3]) or ([200,200,6])
# output: the matrix p (3*3) or (6,6), averaged over pixels for one image.
#        type of transformation
def getPixelMatrixP(origin, transformed, displacement, transformType):

    imgSize = origin.shape[0]
    amountClass = origin.shape[2]
    #print(imgSize)
    #print(amountClass)

    # for each pixel in original image, get corresponding pixel in the transformed image
    # get their class distribution: phi_origin, phi_transformed

    matrix = torch.zeros(amountClass, amountClass)

    for h in range(0, imgSize):

        for w in range (0, imgSize):

            #print('pixel: ' + str((h,w)))
            phi_origin = origin[h][w]

            new_h, new_w  = h,w
            if(transformType == 'flip'):
                new_h, new_w = flipCoordinate((h,w))
            new_h, new_w = getDisplacement((new_h, new_w), displacement)
            phi_transformed = transformed[new_h][new_w]

            ## get matrix for these 2 pixels
            matrix_hw = torch.outer(phi_origin, phi_transformed)

            matrix = matrix + matrix_hw
            #print(matrix_hw)

    #print('total')
    #print(matrix)
    amountPixel = imgSize*imgSize
    #print(matrix/ amountPixel)

    return matrix/amountPixel


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
    if(new_h < 0 or new_h > 2):
        new_h = h
    if (new_w < 0 or new_w > 2):
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



######################### Model ################################


class SegmentationNet10aTrunk(VGGTrunk):
  def __init__(self, cfg):
    super(SegmentationNet10aTrunk, self).__init__()

    self.batchnorm_track = False
    self.conv_size = 3
    self.pad = 1
    self.cfg = cfg
    #self.in_channels = config.in_channels if hasattr(config, 'in_channels') \
    #  else 3
    self.in_channels = 4

    self.features = self._make_layers()

  def forward(self, x):
    x = self.features(x)  # do not flatten
    return x


class SegmentationNet10aHead(nn.Module):
  def __init__(self, output_k, cfg):
    super(SegmentationNet10aHead, self).__init__()

    self.batchnorm_track = False

    self.cfg = cfg
    num_features = self.cfg[-1][0]
    self.num_sub_heads = 1

    self.heads = nn.ModuleList([nn.Sequential(
      nn.Conv2d(num_features, output_k, kernel_size=1,
                stride=1, dilation=1, padding=1, bias=False),
      nn.Softmax2d()) for _ in range(self.num_sub_heads)])

    self.input_sz = (200,200)

  def forward(self, x):
    results = []
    for i in range(self.num_sub_heads):
      x_i = self.heads[i](x)
      x_i = F.interpolate(x_i, size=self.input_sz, mode="bilinear")
      results.append(x_i)

    return results


class SegmentationNet10a(VGGNet):
  cfg = [(64, 1), (128, 1), ('M', None), (256, 1), (256, 1),
         (512, 2), (512, 2)]         # 30x30 recep field

  def __init__(self):
    super(SegmentationNet10a, self).__init__()

    self.batchnorm_track = False

    self.trunk = SegmentationNet10aTrunk(cfg=SegmentationNet10a.cfg)
    self.head = SegmentationNet10aHead(output_k = 6,
                                       cfg=SegmentationNet10a.cfg)

    self._initialize_weights()

  def forward(self, x):
    x = self.trunk(x)
    x = self.head(x)
    return x

def try_gpu():
    """
    If GPU is available, return torch.device as cuda:0; else return torch.device
    as cpu.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('cuda available')
    else:
        device = torch.device('cpu')
        print('cuda unavailable')
    return device

# Temperal,
def feedForward():
    origin_iter, flip_iter, color_iter, crop_iter = getDataIterators()
    # printBatchInfo(origin_iter)

    # Train the model batch by batch
    if (origin_iter.hasNext()):  ##right？  for...

        ## Data for one Batch
        batch_origin = next(origin_iter)
        batch_flip = next(flip_iter)
        batch_color = next(color_iter)
        batch_crop = next(crop_iter)

        print(input.size)

        # when get output
        # getLoss()


def main():

    potsdamData = 'demo/'
    batch_size = 5

    # prepare Original  Dataset, 3 batches, 5 images per batch
    potsdam_origin = Potsdam(root=potsdamData)
    potsdam_origin_loader = torch.utils.data.DataLoader(potsdam_origin, batch_size=batch_size, shuffle=False)
    potsdam_origin_iter = iter(potsdam_origin_loader)

    # prepare Flip Dataset
    potsdam_flip = Potsdam(root=potsdamData, transforms=flip)
    potsdam_flip_loader = torch.utils.data.DataLoader(potsdam_origin, batch_size=batch_size, shuffle=False)
    potsdam_flip_iter = iter(potsdam_origin_loader)

    sampleInput_origin = next(potsdam_origin_iter)
    sampleInput_flip = next(potsdam_flip_iter)

    # create a model
    segmentationModel = SegmentationNet10a().to(try_gpu())

    # get output for different data set
    originOutput = segmentationModel.forward(sampleInput_origin)[0].permute(0,2,3,1)
    flipOutput = segmentationModel.forward(sampleInput_flip)[0].permute(0,2,3,1)
    print(flipOutput.shape)

    # Expect sampleOutput shape: [20, 200, 200, 6]

    # calculate the loss
    loss = getLossT(originOutput, flipOutput)
    print('loss: ' + str(loss))

    loss.backward()
    optimizer = optim.Adam(segmentationModel.parameters(), lr = 0.001)
    optimizer.step()
    print("done")





main()