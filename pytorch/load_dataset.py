from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from utils import load_bangladesh_2015_tiff, load_india_tiff

# from https://discuss.pytorch.org/t/load-tiff-images-to-dataset/8593/3
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
IMG_EXTENSIONS.append('tiff')
IMG_EXTENSIONS.append('tif')


def clean_household_data(csv_file, sat_type):
    """
    - Removes households with zero expenditure
    - Removes households without corresponding tiff
    """
    households = pd.read_csv(csv_file)
    bucket_files = open("../data/bucket_files.txt", "r").readlines()
    bucket_files = [q.split("/")[-1].strip() for q in bucket_files]
    exists = households["a01"].apply(lambda z: "{}_median_{}_{}_500x500_{:.1f}.tif".format(sat_type, "bangladesh", "vis", z) in bucket_files)
    nonzero_exp = households["totexp_m"] > 0
    households = households[np.logical_and(exists, nonzero_exp)]
    households = households.reset_index()
    return households


class BangladeshDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None, sat_type="l8", year=2015):
        """
        Args:
         - csv_file (string): Path to the csv file with annotations.
         - root_dir (string): Directory with all the images.
         - transform (callable, optional): Optional transform to be applied
                on a sample.
         - sat_type (string): l8 or s1
         - year (int): 2011 or 2015
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.sat_type = sat_type
        self.year = year

        self.households = clean_household_data(csv_file, sat_type)

    def __len__(self):
        return len(self.households)


    """
    for tiffs in file
    ---------------------------------------------
    """
    def __getitem__(self, idx):
        hhid = self.households["a01"][idx]
        prefix = self.sat_type
        imgtype = "vis"
        # numpy array, image.shape = (3, 500, 500)
        if prefix == "s1" or self.year == 2015:
            image = load_bangladesh_2015_tiff(self.root_dir, hhid, prefix, imgtype, quiet=True)
        elif prefix == "l8" and self.year == 2011:
            image = load_bangladesh_2011_tiff(self.root_dir, hhid, imgtype, quiet=True)

        # transpose makes shape image.shape = (500, 500, 3)
        image = Image.fromarray(image.transpose((1, 2, 0)))
        expenditure = self.households["totexp_m_pc"][idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            expenditure = self.target_transform(expenditure)

        return image, expenditure


class BangladeshDatasetJpegs(Dataset):

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

    def __len__(self):
        return len(self.households)


    """
    for jpegs in file
    ---------------------------------------------
    """
    def __getitem__(self, idx):
        hhid = self.households["a01"][idx]
        hhid = str(hhid).replace('.', '-')
        img_name = os.path.join(self.root_dir, hhid + '.jpg')
        image = Image.open(img_name)
        #image = io.imread(img_name)
        # TODO: set expenditure index
        expenditure = self.households["totexp_m_pc"][idx]


        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            expenditure = self.target_transform(expenditure)

        return image, expenditure


class IndiaDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None, sat_type="s1", year=2015):
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
        nonzero_exp = self.df["secc_cons_per_cap_scaled"] > 0
        self.df = self.df[nonzero_exp]
        self.df = self.df.reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        hhid = self.df["id"][idx]
        image = load_india_tiff(self.root_dir, hhid, self.sat_type, "vis", quiet=True)
        image = Image.fromarray(image.transpose((1, 2, 0)))
        expenditure = self.df["secc_cons_per_cap_scaled"][idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            expenditure = self.target_transform(expenditure)

        return image, expenditure

