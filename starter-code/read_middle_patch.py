"""
Author:  Anthony Perez
"""

import argparse
import time

import gdal
import numpy as np
from scipy.misc import imshow, imresize

def read(tif_path, H, W):
    '''
    Reads the middle HxW image from the tif given by tif_path
    '''
    gdal_dataset = gdal.Open(tif_path)
    # x_size and y_size and the width and height of the entire tif in pixels
    x_size, y_size = gdal_dataset.RasterXSize, gdal_dataset.RasterYSize
    print("TIF Size (W, H): ", x_size, y_size)
    # Mid point minus half the width and height we want to read will give us our top left corner
    if W > x_size:
        raise Exception("Requested width exceeds tif width.")
    if H > y_size:
        raise Exception("Requested height exceeds tif height.")
    gdal_result = gdal_dataset.ReadAsArray((x_size - W)//2, (y_size - H)//2, W, H)
    # If a tif file has only 1 band, then the band dimension will be removed.
    if len(gdal_result.shape) == 2:
        gdal_result = np.reshape(gdal_result, [1] + list(gdal_result.shape))
    # gdal_result is a rank 3 tensor as follows (bands, height, width)
    return np.transpose(gdal_result, (1, 2, 0))

if __name__ == "__main__":
    default_path = "/afs/ir/class/cs325b/gdal_tutorial/data/F182013.v4c_web.stable_lights.avg_vis.tif"
    parser = argparse.ArgumentParser(description="Read the middle tile from a tif.")
    parser.add_argument('-p', '--tif_path',
        default=default_path,
        help='The path to the tif')
    parser.add_argument('-w','--width', default=1000, type=int, help='Tile width')
    parser.add_argument('-t','--height', default=5000, type=int, help='Tile height')
    args = parser.parse_args()

    start_time = time.time()
    img = read(args.tif_path, args.height, args.width)
    img_time = time.time() - start_time
    start_time = time.time()
    img2 = read(args.tif_path, args.height, args.width*10)
    img2_time = time.time() - start_time
    print("The image you requested took {} seconds to read, an image with ten times the width to {} seconds to read".format(
        img_time, img2_time))

    if args.tif_path == default_path:
        # Something to make the nightlights look good in images
        def clean(x):
            x = np.log(x + 1)
            x[x > 3] = 3
            return x
        img = clean(img)
        img2 = clean(img2)

    img = imresize(np.squeeze(img), 0.25)
    img2 = imresize(np.squeeze(img2), 0.125)

    imshow(img)
    imshow(img2)
