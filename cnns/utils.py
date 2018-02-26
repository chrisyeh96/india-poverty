from __future__ import print_function, division

import time
import os
import gdal
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets, transforms


gdal.SetCacheMax(2**30) # 1 GB

def load_bangladesh_2015_tiff(root_dir, hhid, prefix="s1", imgtype="vis", quiet=True):
    """
    hhid:    household index as float [pull from bangladesh_2015 csv]
    prefix:  either "s1" or "l8"
    imgtype: either "vis" or "multiband"
    """
    source_tiff = "{}/{}_median_bangladesh_{}_500x500_{:.1f}.tif".format(root_dir, prefix, imgtype, hhid)
    if not quiet:
        print("Loading {}...".format(source_tiff))
    gdal_tif = gdal.Open(source_tiff)
    if gdal_tif is None:
        print("IN LOAD 2015 TIFF, GDAL TIF IS NONE")
        return None
    return gdal_tif.ReadAsArray().astype("uint8")


def load_india_tiff(root_dir, hhid, prefix="s1", imgtype="vis", quiet=True):
    """
    hhid:    household index as float [pull from bangladesh_2015 csv]
    prefix:  either "s1" or "l8"
    imgtype: either "vis" or "multiband"
    """
    source_tiff = "{}/{}_median_india_{}_500x500_{:.1f}.tif".format(root_dir, prefix, imgtype, hhid)
    if not quiet:
        print("Loading {}...".format(source_tiff))
    gdal_tif = gdal.Open(source_tiff)
    if gdal_tif is None:
        return None
    return gdal_tif.ReadAsArray().astype("uint8")


def load_bangladesh_2011_tiff(root_dir, hhid, imgtype="vis", quiet=True):
    """
    hhid:    household index as float [pull from bangladesh_2011 csv]
    prefix:  either "s1" or "l8"
    imgtype: either "vis" or "multiband"
    """
    source_tiff = "{}/l8_median_bangladesh_2011_{}_500x500_{:.1f}.tif".format(root_dir, imgtype, hhid)
    if not quiet:
        print("Loading {}...".format(source_tiff))
    gdal_tif = gdal.Open(source_tiff)
    if gdal_tif is None:
        print("IN LOAD 2011 TIFF, GDAL TIF IS NONE")
        return None
    return gdal_tif.ReadAsArray().astype("uint8")

