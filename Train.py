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
batch_size = 11

potsdamData = r'/content/drive/MyDrive/Colab Notebooks/imgs/'
#potsdamData = 'demo/'
#batch_size = 5
EPS = float_info.epsilon


######################### Build Model ################################


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
        outputOrigin = segModel.forward(originBatch, head='A')[0]
        outputOrigin_overCluster = segModel.forward(originBatch, head='B')[0]

        outputFlip = segModel.forward(flipBatch, head='A')[0]
        outputFlip_overCluster = segModel.forward(flipBatch, head='B')[0]

        outputJit = segModel.forward(jitterBatch, head='A')[0]
        outputJit_overCluter = segModel.forward(jitterBatch, head='B')[0]

        # flip back the images for loss calculation
        for i in range(outputFlip.shape[0]):
            img = outputFlip[i]  # [c,200,200], we want to flip it by the last dimension
            flipImg = torchvision.transforms.functional.hflip(img)
            outputFlip[i] = flipImg

        for i in range(outputFlip_overCluster.shape[0]):
            img = outputFlip_overCluster[i]  # [c,200,200], we want to flip it by the last dimension
            flipImg = torchvision.transforms.functional.hflip(img)
            outputFlip_overCluster[i] = flipImg

        #calculate loss and avg over them
        lossFlip, _ = IID_segmentation_loss(outputOrigin, outputFlip)
        lossColor, _ = IID_segmentation_loss(outputOrigin, outputJit)

        lossFlipOver, _ = IID_segmentation_loss(outputOrigin_overCluster, outputFlip_overCluster)
        lossJitOver, _ = IID_segmentation_loss(outputOrigin_overCluster, outputJit_overCluter)

        loss = (lossFlip + lossColor + lossFlipOver + lossJitOver) / 4
        print(loss)

            # Do back propagation
        loss.backward()
        optimizer = optim.Adam(segModel.parameters(), lr = 0.001)
        optimizer.step()

        batch = batch + 1
        print("Done batch " + str(batch))
        #break


    # The model is trained
    # get testing data:
    print('Start Testing')
    potsdamTest_loader = PotsdamTestData.getTestData(batch_size)

    accuracy = 0
    count = 0

    for data in potsdamTest_loader:
        testData = data[0]
        testGt = data[1]
        #print(imgs.shape)
        #print(gts.shape)

        testOutput = segModel.forward(testData, head='A')[0]
        acc = test_accuracy.calculate_accuracy_all(testOutput, testGt, n)
        accuracy = accuracy + acc
        count = count + 1
        print('test batch: ' + str(count))

    accuracy_final = accuracy/count
    print('Test Done')

    print(accuracy_final)






main()