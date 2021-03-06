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
import scipy.io as scio
#import matplotlib.pyplot as plt

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


# specify the path to Ground Truth, and the images file
#gtRoot = 'TestDemoGt/'
#imgRoot = 'demo/'

# On Colab
gtRoot = r'/content/drive/MyDrive/Colab Notebooks/gt/'
imgRoot = r'/content/drive/MyDrive/Colab Notebooks/imgs/'



class Potsdam_test(VisionDataset):

    def __init__(self, root):
        #super()._init_(root, transforms)
        self.root = root
        self.images = os.listdir(root)
        self.groundTruth = os.listdir(root)

    def __getitem__(self, index):

        gt_path = self.root + self.images[index]
        img_path = imgRoot + self.images[index]

        gt = scipy.io.loadmat(gt_path)['gt']#.astype(np.uint8)
        image = scipy.io.loadmat(img_path)['img'].astype(np.uint8)

        # to correct format for model input
        img = Image.fromarray((image))
        imgTensor = torchvision.transforms.functional.pil_to_tensor(img).float()
        gtTensor = torch.Tensor(gt)


        # img [4, 200, 200], gt[200,200]
        return  (imgTensor.float().to('cuda'), gtTensor)
        #return (imgTensor.float(), gtTensor)

    def __len__(self):
        return len(self.images)

def getTestData(batch_size):

    potsdamTestData = Potsdam_test(root=gtRoot)
    potsdamTest_loader = torch.utils.data.DataLoader(potsdamTestData, batch_size=batch_size, shuffle=False)
    print('Test Set size: ' + str(len(potsdamTestData)))

    return potsdamTest_loader




        
