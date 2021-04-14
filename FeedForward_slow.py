import os
from sys import float_info

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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vgg import VGGTrunk, VGGNet
import PotsdamTestData
import test_accuracy

# Note
# This is our original reproduction, which is too slow to run on the actual data.
# Here we run it on a demo data file, which consist of 16 images.
# As for the actual experiments, Train.py is used. Where the loss function from the original IIC repo is used.


# We reproduced the code strictly by the equations presented in the IIC paper - mainly equation 5.
# Namely we perform pixel by pixel comparison (outer product two vector to a matrix) then take the average.         getPixelMatrixP()
# Then we compute such matrix for each image in one batch and take the average.                                     getImageAvgMatrix()
# Each batch consist of 2 transformation, flip and color jitter, so the same is done for each transformation.       getLoss()
# Then we calculate the loss for the matrix we have so far.                                                         getInformation()
# There are 8 displacements, calculation above needs to be done for each of them, and take average.                 getLoss()
# In feedforward function the whole training loop is presented, however the back prob part is commented out since
# It will keep running 'forever'.   By moving to GPU we can see the traning loop works for this small sample data.  feedforward()


#potsdamData = 'demo/'
potsdamData = r'/content/drive/MyDrive/Colab Notebooks/imgs/'
#n = 3  #for potsdam-3
#batch_size = 75
n = 6 #for potsdam-6
batch_size = 11
#displacements = ['up']#,'down','left','right','upright','upleft','downright','downleft']
displacements = ['upright','downleft']
EPS = float_info.epsilon







######################################## Calculate Loss #################################
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

# Equation 5, objective function
# for each displacement, calculate the matrix, which is averaged over 2 transformations
# calculate the loss, then average over the displacements.
def getLoss(output_origin, output_flip, output_color):

    totalLoss = 0

    for displacement in displacements:

        # There are 8 displacements
        print('Displacement: ' + str(displacement))

        print('Transformation: flip')
        transform1 = getImageAvgMatrix(output_origin, output_flip, displacement, 'flip')
        print('Transformation: colorJitter \n')
        transform2 = getImageAvgMatrix(output_origin, output_color, displacement, 'color')

        transformAvgMatrix = (transform1 + transform2) / 2;
        loss = getInformation(transformAvgMatrix)
        totalLoss = totalLoss + loss

    avgLoss = totalLoss / len(displacements)

    return avgLoss


# input: a batch of origin images,
#        a batch of transformed image
#        displacement type(string)
#        type of transformation
# output: 3*3 (potsdam few) or 6*6(potsdam full) matrix, avg over all images in batch.
def getImageAvgMatrix(originBatch, transformedBatch, displacement,transformType):

    amountImage = originBatch.shape[0]  # mostly equals to batchsize but the last one
    amountClass = originBatch.shape[3]
    matrix = torch.zeros(amountClass, amountClass).to('cuda')

    for id in range(amountImage):

        image_x = originBatch[id]
        image_gx = transformedBatch[id]

        imageMatrix = getPixelMatrixP(image_x, image_gx, displacement, transformType)
        matrix = torch.add(matrix, imageMatrix)

    imageAvgMatrix = torch.div(matrix, amountImage)

    return imageAvgMatrix



# input: distribution of images: Φ（x) , Φ（gx）, shape ([200,200,3]) or ([200,200,6])
# or [200, 200, 12], [200,200,24] for over clustering
# output: the matrix p (3*3) or (6,6), averaged over pixels for one image.
#        type of transformation
def getPixelMatrixP(origin, transformed, displacement, transformType):

    imgSize = origin.shape[0]
    amountClass = origin.shape[2]
    #print(imgSize)
    #print(amountClass)

    # for each pixel in original image, get corresponding pixel in the transformed image
    # get their class distribution: phi_origin, phi_transformed

    matrix = torch.zeros(amountClass, amountClass).to('cuda')

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
    if(new_h < 0 or new_h > 199):
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

# input: a batch of flipped images
def flipBatch(flippedBatch):

    for i in range(flippedBatch.shape[0]):
        img = flippedBatch[i]  # [c,200,200], we want to flip it by the last dimension
        flipImg = torchvision.transforms.functional.hflip(img)
        flippedBatch[i] = flipImg

    return flippedBatch

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

######################### Create Model  ################################


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
    self.head = SegmentationNet10aHead(output_k = n, # defined above, 3 or 6
                                       cfg=SegmentationNet10a.cfg)

    self._initialize_weights()

  def forward(self, x):
    x = self.trunk(x)
    x = self.head(x)
    return x

class SegmentationNet10aTwoHead(VGGNet):

      cfg = [(64, 1), (128, 1), ('M', None), (256, 1), (256, 1),
             (512, 2), (512, 2)]

      def __init__(self):
          super(SegmentationNet10aTwoHead, self).__init__()

          self.batchnorm_track = False

          self.trunk = SegmentationNet10aTrunk(cfg = SegmentationNet10a.cfg)
          self.head_A = SegmentationNet10aHead(output_k=n,
                                               cfg=SegmentationNet10a.cfg)
          self.head_B = SegmentationNet10aHead(output_k=4*n,
                                               cfg=SegmentationNet10a.cfg)

          self._initialize_weights()

      def forward(self, x, head="A"):
          x = self.trunk(x)
          if head == "A":
              x = self.head_A(x)
          elif head == "B":
              x = self.head_B(x)
          else:
              assert (False)

          return x


# Temperal,
def feedForward():

    #create a model
    segModel = SegmentationNet10aTwoHead().to(try_gpu())
    print("Model buildt")

    potsdam = Potsdam(root=potsdamData)
    potsdam_loader = torch.utils.data.DataLoader(potsdam, batch_size=batch_size, shuffle=True)

    # Train the model batch by batch
    batch = 1
    for data in potsdam_loader:
        print('########## Batch ' + str(batch) + ' ######## '+ str(data.shape[0]) + ' images ##########')

        originList = []
        flipList = []
        jitterList =[]

        for i in range (data.shape[0]):

            #Unpack 'data', which consist of 3 tensors for each image
            #Make dataset for one batch: original, flipped and colorjittered
            origin = data[i][0]
            flip = data[i][1]
            jitter = data[i][2]

            originList.append(origin)
            flipList.append(flip)
            jitterList.append(jitter)



        #shape: batchSize, 4, 200,200
        originBatch = torch.stack(originList)
        flipBatch = torch.stack(flipList)
        jitterBatch = torch.stack(jitterList)

        # get output distribution for different data set : Original, flipped and colorjittered image data
        # also for over-clustering
        outputOrigin = segModel.forward(originBatch, head='A')[0].permute(0,2,3,1)
        outputOrigin_overCluster = segModel.forward(originBatch, head='B')[0].permute(0, 2, 3, 1)

        outputFlip = segModel.forward(flipBatch, head='A')[0].permute(0,2,3,1)
        outputFlip_overCluster = segModel.forward(flipBatch, head='B')[0].permute(0, 2, 3, 1)

        outputJit = segModel.forward(jitterBatch, head='A')[0].permute(0,2,3,1)
        outputJit_overCluter = segModel.forward(jitterBatch, head='B')[0].permute(0,2,3,1)

        # calculate avg loss
        loss = getLoss(outputOrigin, outputFlip,outputJit)
        print('Over-clustering')
        loss_overCluster = getLoss(outputOrigin_overCluster, outputFlip_overCluster,outputJit_overCluter)
        avgLoss = (loss + loss_overCluster)/2

        print('loss: ' + str(loss))
        print('overCluster loss: ' + str(loss_overCluster))
        print('avg loss: ' + str(avgLoss))

        #### back propagation to train the model ###########
        avgLoss.backward()
        optimizer = optim.Adam(segModel.parameters(), lr = 0.001)
        optimizer.step()

        print("Done batch " + str(batch))

        #break
        batch = batch + 1

    print('Total: '+ str(batch) + ' batches')

    # The model is trained
    # get testing data:
    print('Start Testing')
    potsdamTest_loader = PotsdamTestData.getTestData(batch_size)

    accuracy = 0
    count = 0

    for data in potsdamTest_loader:
        testData = data[0]
        testGt = data[1]
        # print(imgs.shape)
        # print(gts.shape)

        testOutput = segModel.forward(testData, head='A')[0]
        acc = test_accuracy.calculate_accuracy_all(testOutput, testGt, n)
        accuracy = accuracy + acc
        count = count + 1
        print('test batch: ' + str(count))

    accuracy_final = accuracy / count
    print('Test Done')

    print(accuracy_final)


def main():

    feedForward()


main()