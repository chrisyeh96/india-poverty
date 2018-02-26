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


class BangladeshDataset(Dataset):

  def __init__(self, csv_file, root_dir, transform=None, target_transform=None, sat_type="l8", year=2015,
         use_grouped_labels=False, village=None):
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
    self.village = village
    if use_grouped_labels:
      self.grouped_labels = self.households.groupby("Village")["totexp_m_pc"].mean()
    if self.village:
      self.households = self.households[self.households["Village"] == village]
      self.households = self.households.reset_index()

  def __len__(self):
    return len(self.households)

  def __getitem__(self, idx):
    hhid = self.households["a01"][idx]
    prefix = self.sat_type
    imgtype = "vis"

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

    if self.use_grouped_labels:
      return image, expenditure, village
    else:
      return image, expenditure

  def get_grouped_labels(self):
    return self.grouped_labels

  def get_households(self):
    return self.households


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
    expenditure = self.households["totexp_m_pc"][idx]
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      expenditure = self.target_transform(expenditure)

    return image, expenditure


class IndiaDataset(Dataset):

  def __init__(self, csv_file, root_dir, label,
         transform=None, target_transform=None,
         sat_type="s1", year=2015, frac=1.0):
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
    self.sat_type = sat_type
    self.df = self.df[~np.isnan(self.df[self.label])]
    self.df = self.df.reset_index()
    if frac < 1:
      self.df = self.df.sample(frac=frac)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    hhid = self.df["id"][idx]
    image = load_india_tiff(self.root_dir, hhid, self.sat_type, "vis", quiet=True)
    image = Image.fromarray(image.transpose((1, 2, 0)))
    expenditure = self.df[self.label][idx]
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      expenditure = self.target_transform(expenditure)
    return image, expenditure

