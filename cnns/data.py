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

  def __init__(self, csv_file, root_dir, label="totexp_m_pc",
               transform=None, target_transform=None,
               sat_type="s1", year=2015):
    self.df = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform
    self.target_transform = target_transform
    self.sat_type = sat_type
    self.year = year
    self.label = label

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    hhid = self.df["a01"][idx]
    if self.year == 2015:
      image = load_bangladesh_2015_tiff(self.root_dir, hhid, self.sat_type, "vis", quiet=True)
    elif self.year == 2011:
      image = load_bangladesh_2011_tiff(self.root_dir, hhid, self.sat_type, "vis", quiet=True)
    image = Image.fromarray(image.transpose((1, 2, 0)))
    expenditure = self.df[self.label][idx]
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

