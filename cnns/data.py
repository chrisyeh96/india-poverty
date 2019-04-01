from __future__ import print_function, division

import gdal
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


gdal.SetCacheMax(2**30) # 1 GB


class IndiaDataset(Dataset):

    def __init__(self, csv_file, root_dir, label,
                 transform=None, target_transform=None,
                 year=2015, frac=1.0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.target_transform = target_transform
        self.df = self.df[~np.isnan(self.df[self.label])]
        if frac < 1:
            self.df = self.df.sample(frac=frac, replace=False)
        self.df = self.df.reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        hhid = self.df["id"][idx]
        s1 = load_india_tiff(self.root_dir, hhid, "s1", "vis", quiet=True)
        l8 = load_india_tiff(self.root_dir, hhid, "l8", "vis", quiet=True)
        s1 = Image.fromarray(s1.transpose((1, 2, 0)))
        l8 = Image.fromarray(l8.transpose((1, 2, 0)))
        expenditure = self.df[self.label][idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            expenditure = self.target_transform(expenditure)
        return image, expenditure


def load_india_tiff(root_dir, village_id, prefix="s1", imgtype="vis", quiet=True):
    """
    Load India tiff.
    """
    source_tiff = f"{root_dir}/{prefix}_median_india_{imgtype}_500x500_{village_id:.1f}.tif"
    if not quiet:
        print("Loading {}...".format(source_tiff))
    gdal_tif = gdal.Open(source_tiff)
    if gdal_tif is None:
        return None
    return gdal_tif.ReadAsArray().astype("uint8")
