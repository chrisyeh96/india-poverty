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
from utils import load_tiff_bangladesh, load_tiff_india

# from https://discuss.pytorch.org/t/load-tiff-images-to-dataset/8593/3
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
IMG_EXTENSIONS.append('tiff')
IMG_EXTENSIONS.append('tif')

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")


class BangladeshDataset(Dataset):
    """Bangladesh Poverty dataset."""

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None, sat_type="l8"):
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
        self.sat_type = sat_type
        bucket_files = open("../data/bucket_files.txt", "r").readlines()
        bucket_files = [q.split("/")[-1].strip() for q in bucket_files]
        exists = self.households["a01"].apply(lambda z: "{}_median_{}_{}_500x500_{:.1f}.tif".format(sat_type, "bangladesh", "vis", z) in bucket_files)
        nonzero_exp = self.households["totexp_m"] > 0
        self.households = self.households[np.logical_and(exists, nonzero_exp)]
        self.households = self.households.reset_index()

    def __len__(self):
        return len(self.households)

    def __getitem__(self, idx):
        # TODO: make sure 0th index is jpeg file
        hhid = self.households["a01"][idx]
        prefix = self.sat_type
        imgtype = "vis"
        # numpy array, image.shape = (3, 500, 500)
        image = load_tiff_bangladesh(self.root_dir, hhid, prefix, imgtype, quiet=True)
        # transpose makes shape image.shape = (500, 500, 3)
        image = Image.fromarray(image.transpose((1, 2, 0)))
        #hhid = str(hhid).replace('.', '-')
        # TODO: should transform to PIL image?
        #image = Image.open(img_name)
        # TODO: set expenditure index
        expenditure = self.households["totexp_m"][idx]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            expenditure = self.target_transform(expenditure)

        return image, expenditure


class IndiaDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None, sat_type="s1"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.sat_type = sat_type
#        bucket_files = open("../data/staff_bucket_files.txt", "r").readlines()
#        bucket_files = [q.split("/")[-1].strip() for q in bucket_files]
#        exists = self.df["id"].apply(lambda z: "{}_median_{}_{}_500x500_{:.1f}.tif".format(sat_type, "india", "vis", z) in bucket_files)
        nonzero_exp = self.df["secc_cons_per_hh"] > 0
#        self.df = self.df[np.logical_and(exists, nonzero_exp)]
        self.df = self.df[nonzero_exp]
        self.df = self.df.reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # TODO: make sure 0th index is jpeg file
        hhid = self.df["id"][idx]
        # numpy array, image.shape = (3, 500, 500)
        image = load_tiff_india(self.root_dir, hhid, self.sat_type, "vis", quiet=True)
        # transpose makes shape image.shape = (500, 500, 3)
        image = Image.fromarray(image.transpose((1, 2, 0)))
        #hhid = str(hhid).replace('.', '-')
        # TODO: should transform to PIL image?
        #image = Image.open(img_name)
        # TODO: set expenditure index
        expenditure = self.df["secc_cons_per_hh"][idx]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            expenditure = self.target_transform(expenditure)

        return image, expenditure

