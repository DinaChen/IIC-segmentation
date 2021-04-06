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
import matplotlib.pyplot as plt

def Save_image_with_gt():
    folder_gt = 'gt/'
    folder_imgs= 'imgs/'
    new_folder= 'imgs_with_gt/'
    
    for imageId in os.listdir(folder_gt):
        path = folder_imgs + imageId
        shutil.copy(path, new_folder+imageId)     
        
     
Save_image_with_gt()

