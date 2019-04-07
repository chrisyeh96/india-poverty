from __future__ import print_function, division

import gdal
import numpy as np
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

gdal.SetCacheMax(2 ** 30) # 1 GB


sat_transforms = {
    "l8": [transforms.CenterCrop(100)],
    "s1": [transforms.CenterCrop(300)],
    "train": [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ],
    "val": [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ],
}


def get_dataloader(csv_path, root_dir, label, batch_size=128, train=True, frac=1.0):
    dataset = IndiaDataset(csv_file=csv_path, root_dir=root_dir, label=label,
                           train=train, frac=frac)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8,
                        shuffle=train)
    return loader


class IndiaDataset(Dataset):

    def __init__(self, csv_file, root_dir, label, train=False, frac=1.0):
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
        self.train = train
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
        expenditure = self.df[self.label][idx].astype(np.float32)
        if self.train:
            s1_transform = transforms.Compose(sat_transforms["s1"] +
                                              sat_transforms["train"])
            l8_transform = transforms.Compose(sat_transforms["l8"] +
                                              sat_transforms["train"])
        else:
            s1_transform = transforms.Compose(sat_transforms["s1"] +
                                              sat_transforms["val"])
            l8_transform = transforms.Compose(sat_transforms["l8"] +
                                              sat_transforms["val"])
        s1, l8 = s1_transform(s1), l8_transform(l8)
        return torch.cat((s1, l8), dim=0), expenditure


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
