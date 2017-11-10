from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
#from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from geotiling import GeoProps
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import gdal

gdal.SetCacheMax(2**30) # 1 GB

######################################################################
# Load tif from hhid as numpy array.
# This helper function is specific to 2015 Bangladesh tifs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# E.g. l8_median_bangladesh_vis_500x500_1000.0.tif
# E.g. s1_median_bangladesh_vis_500x500_1000.0.tif
#
# Parameters:
#   - hhid: household id. Float
#   - prefix: "s1" or "l8" corresponding to Sentinel-1 or Landsat8. String
#   - imgtype: "vis" or "multiband". String
#   - quiet:
#
# Returns:
#   - gdal_tif: numpy array corresponding to gdal tif
#

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


######################################################################
# Load tif from hhid as numpy array.
# This helper function is specific to 2011 Bangladesh tifs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# E.g. l8_median_bangladesh_2011_multiband_500x500_1000.0.tif
# E.g. l8_median_bangladesh_2011_vis_500x500_1000.0.tif
#
# Parameters:
#   - hhid: household id. Float
#   - prefix: "s1" or "l8" corresponding to Sentinel-1 or Landsat8. String
#   - imgtype: "vis" or "multiband". String
#   - quiet:
#
# Returns:
#   - gdal_tif: numpy array corresponding to gdal tif
#

def load_bangladesh_2011_tiff(root_dir, hhid, imgtype="vis", quiet=True):
    """
    hhid:    household index as float [pull from bangladesh_2011 csv]
    prefix:  either "s1" or "l8"
    imgtype: either "vis" or "multiband"
    """
    #source_tiff = "/mnt/staff-bucket/{}_median_bangladesh_{}_500x500_{:.1f}.tif".format(prefix, imgtype, hhid)
    source_tiff = "{}/l8_median_bangladesh_2011_{}_500x500_{:.1f}.tif".format(root_dir, imgtype, hhid)
    if not quiet:
        print("Loading {}...".format(source_tiff))
    gdal_tif = gdal.Open(source_tiff)
    if gdal_tif is None:
        print("IN LOAD 2011 TIFF, GDAL TIF IS NONE")
        return None
    return gdal_tif.ReadAsArray().astype("uint8")





######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, dataloders, use_gpu, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels.float())

        outputs = model(inputs)
        preds = outputs.data

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            #ax.set_title('predicted: {}'.format(preds[j]))
            torch_to_im_show(inputs.cpu().data[j])

            if images_so_far == num_images:
                return



######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def torch_to_im_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
