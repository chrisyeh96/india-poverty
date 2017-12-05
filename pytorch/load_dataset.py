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
from skimage import transform as tf

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]



def clean_household_data(csv_file, sat_type):
    """
    - Removes households with zero expenditure
    - Removes households without corresponding tiff
    """
    households = pd.read_csv(csv_file)
    bucket_files = open("../data/bucket_files.txt", "r").readlines()
    bucket_files = [q.split("/")[-1].strip() for q in bucket_files]
    exists = households["a01"].apply(lambda z: "{}_median_{}_{}_500x500_{:.1f}.tif".format(sat_type, "bangladesh", "vis", z) in bucket_files)
    duplicate = households["a01"].apply(lambda z: str(z)[-1] == '0')
    households = households[np.logical_and(exists, duplicate)]
    households = households[pd.notnull(households['totexp_m_pc'])]
    households = households.reset_index()
    return households


class BangladeshDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None, sat_type="l8", year=2015,
                 use_grouped_labels=False):
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
        self.use_grouped_labels = use_grouped_labels
        self.grouped_labels = None
        if use_grouped_labels:
            self.grouped_labels = self.households.groupby("Village")["totexp_m_pc"].mean()

    def __len__(self):
        return len(self.households)

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

        village = self.households["Village"][idx]
        if self.use_grouped_labels:
            village = self.households["Village"][idx]
            expenditure = self.grouped_labels[village]
        else:
            expenditure = self.households["totexp_m_pc"][idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            expenditure = self.target_transform(expenditure)

        return image, expenditure, village
    
    def get_grouped_labels(self):
        return self.grouped_labels




# for 2015 multiband data
class BangladeshMultibandDataset(Dataset):
    """Bangladesh Poverty dataset."""

    def __init__(self, csv_file, root_dir,sat_type="l8",mean=[0,0,0],std=[1,1,1],use_grouped_labels=False,target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.target_transform = target_transform
        self.sat_type = sat_type
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.households = clean_household_data(csv_file, sat_type)
        self.use_grouped_labels = use_grouped_labels
        self.grouped_labels = None
        if use_grouped_labels:
            self.grouped_labels = self.households.groupby("Village")["totexp_m_pc"].mean()

        #########
        #########

        # self.households = pd.read_csv(csv_file)
        # self.root_dir = root_dir
        
        # self.crop_size = crop_size
        # self.mean = np.array(mean)
        # self.std = np.array(std)
        # bucket_files = open("../data/bucket_files.txt", "r").readlines()
        # bucket_files = [q.split("/")[-1].strip() for q in bucket_files]
        # # l8_median_bangladesh_2011_multiband_500x500_11.0.tif 
        # exists = self.households["a01"].apply(lambda z: "{}_median_{}_{}_500x500_{:.1f}.tif".format(sat_type, "bangladesh", "multiband", z) in bucket_files)
        # nonzero_exp = self.households["totexp_m_pc"] > 0
        # self.households = self.households[np.logical_and(exists, nonzero_exp)]
        # self.households = self.households.reset_index()
        
    def __len__(self):
        return len(self.households)


    def __getitem__(self, idx):
        hhid = self.households["a01"][idx]
        prefix = self.sat_type
        imgtype = "multiband"

        image = load_bangladesh_2015_tiff(self.root_dir, hhid, prefix, imgtype, quiet=True)
        # transpose makes shape image.shape = (500, 500, 3) #for multiband 6
        image = image.transpose((1, 2, 0))
        # image = image.astype(np.uint8)
        image = crop_center(image,224,224)
        # image = tf.resize(image, (224, 224, 6), order=0)
        # image_rgb = image[:,:,:3]
        # image_rgb = Image.fromarray(image_rgb)
        # image_nonrgb = image[:,:,3:]
        # image_nonrgb = Image.fromarray(image_nonrgb)


        #### removing this ####
        # image = crop_center(image,self.crop_size,self.crop_size)
        #### removing this ####

        #normalize
        image = (image.astype(np.float32) - self.mean)/self.std
        image = image.transpose((2,0,1))

        village = self.households["Village"][idx]
        if self.use_grouped_labels:
            village = self.households["Village"][idx]
            expenditure = self.grouped_labels[village]
        else:
            expenditure = self.households["totexp_m_pc"][idx]


        if self.target_transform:
            expenditure = self.target_transform(expenditure)

        
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     expenditure = self.target_transform(expenditure)

        return image, expenditure,village

    def get_grouped_labels(self):
        return self.grouped_labels





class BangladeshDatasetJpegs(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None, sat_type="l8",
                 use_grouped_labels=False):
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

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None, 
                 sat_type="s1", year=2015, use_grouped_labels=False):
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
        self.df = self.df[~np.isnan(self.df["secc_cons_per_cap_scaled"])]
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

