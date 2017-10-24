import os
import gdal
import numpy as np
import pandas as pd
from geotiling import GeoProps
from scipy.misc import imsave
import matplotlib.pyplot as plt

gdal.SetCacheMax(2**30) # 1 GB

bangladesh_2011 = pd.read_csv("/mnt/mounted_bucket/Bangladesh_CE_2011.csv")
bangladesh_2015 = pd.read_csv("/mnt/mounted_bucket/Bangladesh_CE_2015.csv")

bangladesh_2015_train = pd.read_csv("/mnt/mounted_bucket/bangladesh_2015_train.csv")
bangladesh_2015_valid = pd.read_csv("/mnt/mounted_bucket/bangladesh_2015_valid.csv")

data_path = '/home/echartock03/data/bangladesh_vis_jpgs'

def load_tiff(hhid, prefix="s1", imgtype="vis", quiet=False):
    """
    hhid:    household index as float [pull from bangladesh_2015 csv]
    prefix:  either "s1" or "l8"
    imgtype: either "vis" or "multiband"
    """
    source_tiff = "/mnt/mounted_bucket/{}_median_bangladesh_{}_500x500_{:.1f}.tif".format(prefix, imgtype, hhid)
    if not quiet:
        print("Loading {}...".format(source_tiff))
    gdal_tif = gdal.Open(source_tiff)
    if gdal_tif is None:
        return None
    return gdal_tif.ReadAsArray().astype("uint8")


def vis_tif(img):
    plt.imshow(img.transpose(1,2,0))


print "Copying {} tifs to jpgs.".format(len(bangladesh_2015_train))
n_files = 0
for i, hhid in enumerate(bangladesh_2015_train["a01"]):
    prefix = "l8"
    imgtype = "vis"
    img = load_tiff(hhid, prefix, imgtype, True)
    if img is None:
        print "Image {:.1f} cannot be found.".format(hhid)
    else:
        img = img.transpose(1,2,0)
        #vis_tif(img)
        hhid = str(hhid).replace('.', '-')
        img_file = os.path.join(data_path, "train", hhid + ".jpg") 
        imsave(img_file, img)
        if (i + 1) % 100 == 0:
            print "{} jpgs created.".format(i + 1)
        n_files = i
print "{} jpgs created.".format(n_files)
