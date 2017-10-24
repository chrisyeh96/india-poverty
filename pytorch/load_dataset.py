from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")


class BangladeshDataset(Dataset):
    """Bangladesh Poverty dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
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

    def __len__(self):
        return len(self.households)

    def __getitem__(self, idx):
        # TODO: make sure 0th index is jpeg file
        img_name = os.path.join(self.root_dir, self.households.ix[idx, 0])
        image = io.imread(img_name)
        # TODO: set expenditure index
        expenditure = self.households.ix[idx, 1:].as_matrix().astype('float')
        sample = {'image': image, 'expenditure': expenditure}

        if self.transform:
            sample = self.transform(sample)

        return sample