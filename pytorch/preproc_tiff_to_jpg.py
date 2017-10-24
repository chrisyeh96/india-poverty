import gdal
import numpy as np
import pandas as pd
from geotiling import GeoProps

bangladesh_2011 = pd.read_csv("/mnt/mounted_bucket/Bangladesh_CE_2011.csv")
bangladesh_2015 = pd.read_csv("/mnt/mounted_bucket/Bangladesh_CE_2015.csv")

bangladesh_2015_train = pd.read_csv("/mnt/mounted_bucket/bangladesh_2015_train.csv")
bangladesh_2015_valid = pd.read_csv("/mnt/mounted_bucket/bangladesh_2015_valid.csv")


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
  return gdal_tif.ReadAsArray().astype("uint8")


def vis_tif(img):
  plt.imshow(img.transpose(1,2,0))


for i, hhid in enumerate(bangladesh_2015_train["a01"]):
	print type(hhid)
    if '.' not in hhid:
    	hhid += '.0'

    prefix = "l8"
    imgtype = "vis"
	img = load_tiff(hhid, prefix, imgtype, True)
	vis_tif(img)

	# logic to save tif

	if i == 0:
		break