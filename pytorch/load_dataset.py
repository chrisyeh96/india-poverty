from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io #, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# from https://discuss.pytorch.org/t/load-tiff-images-to-dataset/8593/3
#from torchvision.datasets import ImageFolder
#from torchvision.datasets.folder import IMG_EXTENSIONS
#IMG_EXTENSIONS.append('tiff')

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")





class BangladeshDataset(Dataset):
    """Bangladesh Poverty dataset."""

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.households = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.households)

    def __getitem__(self, idx):
        # TODO: make sure 0th index is jpeg file
        hhid = self.households["a01"][idx]
        hhid = str(hhid).replace('.', '-')
        img_name = os.path.join(self.root_dir, hhid + '.jpg')
        image = Image.open(img_name)
        #image = io.imread(img_name)
        # TODO: set expenditure index
        expenditure = self.households["totexp_m"][idx]
        

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            expenditure = self.target_transform(expenditure)

        return image, expenditure
