"""
Author:  Anthony Perez
"""

import sys
import numpy as np
from scipy.misc import imshow, imresize
import gdal
from geotiling import GeoProps

def pixel_loc_to_geo_loc(col, row, is_meters=False,
                         source_tif='/afs/ir/class/cs325b/gdal_tutorial/data/F182013.v4c_web.stable_lights.avg_vis.tif'):
    props = GeoProps()
    gdal_tif = gdal.Open(source_tif)
    props.import_geogdal(gdal_tif)

    lon, lat = props.colrow2lonlat(col, row)
    if is_meters:
        lon, lat = props.get_geocoord(lon, lat)

    return lon, lat

def visualize_location(center_lon, center_lat, image_pixel_size=2000, is_meters=False,
                   source_tif='/afs/ir/class/cs325b/gdal_tutorial/data/F182013.v4c_web.stable_lights.avg_vis.tif'):
    '''
    Reads the image_pixel_size by image_pixel_size image at the longitude center_lon and latitude center_lat
    '''
    # Read the file and load the meta data
    props = GeoProps()
    gdal_tif = gdal.Open(source_tif)
    props.import_geogdal(gdal_tif)

    # Get the locs in pixel coordinates
    if is_meters:
        # Convert (lon, lat) to (horizontal meters, vertical meters)  Note the names become confusing
        # center_lon is now the center horizontal coordinate in meters
        center_lon, center_lat = props.get_affinecoord(center_lon, center_lat)
    # props.lonlat2colrow uses whatever the projetion uses to linearly transform its parameters
    # This means the units of the input to props.lonlat2colrow should match the units of the projection
    center_col, center_row = props.lonlat2colrow(center_lon, center_lat)
    left_col, top_row = center_col - image_pixel_size // 2, center_row - image_pixel_size // 2
    
    print("Quantized (longitude, latitude): ")
    print(pixel_loc_to_geo_loc(center_col, center_row, is_meters=is_meters, source_tif=source_tif))

    # No input checking so this call may return an error (give it a try, use center_lon = 100000.0)
    image = gdal_tif.ReadAsArray(left_col, top_row, image_pixel_size, image_pixel_size)
    if len(image.shape) == 2:
        return image
    image = np.transpose(image, (1,2,0))
    return image

if __name__ == "__main__":
    print("Usage: python visualize_grid.py <center_lon>, <center_lat>")
    center_lon = float(sys.argv[1])
    center_lat = float(sys.argv[2])
    print("center_lon: {}, center_lat: {}".format(center_lon, center_lat))
    image = visualize_location(center_lon, center_lat)

    def clean(x):
	x = np.log(x + 1)
	x[x > 3] = 3
	return x
    image = clean(image)
    image = imresize(image, (500, 500))

    imshow(image)
