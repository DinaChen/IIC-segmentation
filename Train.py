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

# Potsdam-3 and Potsdam-6

#n = 3  #for potsdam-3
#batch_size = 75
n = 6 #for potsdam-6
#batch_size = 60

#potsdamData = r'/content/drive/MyDrive/Colab Notebooks/demo/'
potsdamData = 'demo/'
batch_size = 5
displacements = ['up']#'down','left','right','upright','upleft','downright','downleft']
EPS = float_info.epsilon


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
        print('Displacement: ' + str(displacement))

        transform1 = getImageAvgMatrix(output_origin, output_flip, displacement, 'flip')

        transformAvgMatrix = (transform1 + transform1 + transform1) / 3;
        loss = getInformation(transformAvgMatrix)
        totalLoss = totalLoss + loss

    avgLoss = totalLoss / 8

    return avgLoss

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

    avgLoss = totalLoss / 8

    return avgLoss


# input: a batch of origin images,
#        a batch of transformed image
#        displacement type(string)
#        type of transformation
# output: 3*3 (potsdam few) or 6*6(potsdam full) matrix, avg over all images in batch.
def getImageAvgMatrix(originBatch, transformedBatch, displacement,transformType):

    amountImage = originBatch.shape[0]  # mostly equals to batchsize but the last one
    amountClass = originBatch.shape[3]
    matrix = torch.zeros(amountClass, amountClass)#.to('cuda')

    for id in range(amountImage):
        #print(id + 1)

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

    matrix = torch.zeros(amountClass, amountClass)#.to('cuda')

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


######################### Calculate Loss ################################

def random_translation_multiple(data, half_side_min, half_side_max):
  n, c, h, w = data.shape

  # pad last 2, i.e. spatial, dimensions, equally in all directions
  data = F.pad(data,
               (half_side_max, half_side_max, half_side_max, half_side_max),
               "constant", 0)
  assert (data.shape[2:] == (2 * half_side_max + h, 2 * half_side_max + w))

  # random x, y displacement
  t = np.random.randint(half_side_min, half_side_max + 1, size=(2,))
  polarities = np.random.choice([-1, 1], size=(2,), replace=True)
  t *= polarities

  # -x, -y in orig img frame is now -x+half_side_max, -y+half_side_max in new
  t += half_side_max

  data = data[:, :, t[1]:(t[1] + h), t[0]:(t[0] + w)]
  assert (data.shape[2:] == (h, w))

  return data





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

# Temperal,
def feedForward():

    #create a model
    #segmentationModel = SegmentationNet10a().to(try_gpu())
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

        # avgLoss.backward()
        # optimizer = optim.Adam(segModel.parameters(), lr = 0.001)
        # optimizer.step()

        # print("Done batch " + str(batch))

        break
        batch = batch + 1

    print('Total: '+ str(batch-1) + ' batches')


# IIC original code
# Calculate the loss given two batch of output
def IID_segmentation_loss(x1_outs, x2_outs, all_affine2_to_1=None,
                              all_mask_img1=None, lamb=1.0,
                              half_T_side_dense= 10,
                              half_T_side_sparse_min=0,
                              half_T_side_sparse_max=0):
        assert (x1_outs.requires_grad)
        assert (x2_outs.requires_grad)
        # assert (not all_affine2_to_1.requires_grad)
        # assert (not all_mask_img1.requires_grad)

        assert (x1_outs.shape == x2_outs.shape)

        # bring x2 back into x1's spatial frame
        # todo : do this before this function
        # x2_outs_inv = perform_affine_tf(x2_outs, all_affine2_to_1)

        # Displacement
        if (half_T_side_sparse_min != 0) or (half_T_side_sparse_max != 0):
          x2_outs_inv = random_translation_multiple(x2_outs,
                                                    half_side_min=half_T_side_sparse_min,
                                                    half_side_max=half_T_side_sparse_max)

        # if RENDER:
        # indices added to each name by render()
        #  render(x1_outs, mode="image_as_feat", name="invert_img1_")
        #  render(x2_outs, mode="image_as_feat", name="invert_img2_pre_")
        #  render(x2_outs_inv, mode="image_as_feat", name="invert_img2_post_")
        #  render(all_mask_img1, mode="mask", name="invert_mask_")

        # zero out all irrelevant patches
        bn, k, h, w = x1_outs.shape
        #all_mask_img1 = all_mask_img1.view(bn, 1, h, w)  # mult, already float32
        #x1_outs = x1_outs * all_mask_img1  # broadcasts
        #x2_outs = x2_outs * all_mask_img1

        # sum over everything except classes, by convolving x1_outs with x2_outs_inv
        # which is symmetric, so doesn't matter which one is the filter
        x1_outs = x1_outs.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
        x2_outs = x2_outs.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w

        # k, k, 2 * half_T_side_dense + 1,2 * half_T_side_dense + 1
        p_i_j = F.conv2d(x1_outs, weight=x2_outs, padding=(half_T_side_dense,
                                                           half_T_side_dense))
        p_i_j = p_i_j.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)  # k, k

        # normalise, use sum, not bn * h * w * T_side * T_side, because we use a mask
        # also, some pixels did not have a completely unmasked box neighbourhood,
        # but it's fine - just less samples from that pixel
        current_norm = float(p_i_j.sum())
        p_i_j = p_i_j / current_norm

        # symmetrise
        p_i_j = (p_i_j + p_i_j.t()) / 2.

        # compute marginals
        p_i_mat = p_i_j.sum(dim=1).unsqueeze(1)  # k, 1
        p_j_mat = p_i_j.sum(dim=0).unsqueeze(0)  # 1, k

        # for log stability; tiny values cancelled out by mult with p_i_j anyway
        p_i_j[(p_i_j < EPS).data] = EPS
        p_i_mat[(p_i_mat < EPS).data] = EPS
        p_j_mat[(p_j_mat < EPS).data] = EPS

        # maximise information
        loss = (-p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i_mat) -
                          lamb * torch.log(p_j_mat))).sum()

        # for analysis only
        loss_no_lamb = (-p_i_j * (torch.log(p_i_j) - torch.log(p_i_mat) -
                                  torch.log(p_j_mat))).sum()

        return loss, loss_no_lamb



def main():

    #feedForward()

    segModel = SegmentationNet10aTwoHead().to(try_gpu())
    print("Model buildt")

    potsdam = Potsdam(root=potsdamData)
    potsdam_loader = torch.utils.data.DataLoader(potsdam, batch_size=batch_size, shuffle=True)

    # Train the model batch by batch
    batch = 0
    for data in potsdam_loader:
        print('########## Batch ' + str(batch+1) + ' ######## ' + str(data.shape[0]) + ' images ##########')

        originList = []
        flipList = []
        jitterList = []

        for i in range(data.shape[0]):
            # Unpack 'data', which consist of 3 tensors for each image
            # Make dataset for one batch: original, flipped and colorjittered
            origin = data[i][0]
            flip = data[i][1]
            jitter = data[i][2]

            originList.append(origin)
            flipList.append(flip)
            jitterList.append(jitter)

        # shape: batchSize, 4, 200,200
        originBatch = torch.stack(originList)
        flipBatch = torch.stack(flipList)
        jitterBatch = torch.stack(jitterList)

        # get output distribution
        #outputOrigin = segModel.forward(originBatch, head='A')[0]
        #outputOrigin_overCluster = segModel.forward(originBatch, head='B')[0]

        #outputFlip = segModel.forward(flipBatch, head='A')[0]
        #outputFlip_overCluster = segModel.forward(flipBatch, head='B')[0]

        #outputJit = segModel.forward(jitterBatch, head='A')[0]
        #outputJit_overCluter = segModel.forward(jitterBatch, head='B')[0]

        # flip back the images for loss calculation
        #for i in range(outputFlip.shape[0]):
        #    img = outputFlip[i]  # [c,200,200], we want to flip it by the last dimension
        #    flipImg = torchvision.transforms.functional.hflip(img)
        #    outputFlip[i] = flipImg

        #for i in range(outputFlip_overCluster.shape[0]):
        #    img = outputFlip_overCluster[i]  # [c,200,200], we want to flip it by the last dimension
        #    flipImg = torchvision.transforms.functional.hflip(img)
        #    outputFlip_overCluster[i] = flipImg

        #calculate loss and avg over them
        #lossFlip, _ = IID_segmentation_loss(outputOrigin, outputFlip)
        #lossColor, _ = IID_segmentation_loss(outputOrigin, outputJit)

        #lossFlipOver, _ = IID_segmentation_loss(outputOrigin_overCluster, outputFlip_overCluster)
        #lossJitOver, _ = IID_segmentation_loss(outputOrigin_overCluster, outputJit_overCluter)

        #loss = (lossFlip + lossColor + lossFlipOver + lossJitOver) / 4
        #print(loss)

            # Do back propagation
        # loss.backward()
        # optimizer = optim.Adam(segModel.parameters(), lr = 0.001)
        # optimizer.step()

        batch = batch + 1
        print("Done batch " + str(batch))
        break

    # The model is trained
    # get testing data:
    print('Start Testing')
    testData, testGt = PotsdamTestData.getTestData()
    #print(testData.shape)
    testOutput = segModel.forward(testData, head='A')[0]
    #print(testOutput.shape)
    #print(testGt.shape)

    accuracy = test_accuracy.calculate_accuracy_all(testOutput, testGt, n)
    print(accuracy)






main()