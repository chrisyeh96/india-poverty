import gdal
import numpy as np

# driver parameters
output_path = './test.tif'
num_x_pixels = 200
num_y_pixels = 100
num_bands = 1
raster_data_type = gdal.GDT_Float32

# GeoTransform: these are the lat/lon coordinates for Stanford's Main Quad
x_min_loc = -122.170277
y_min_loc = 37.427491
PIXEL_WIDTH = 0.0083333333
PIXEL_HEIGHT = 0.0083333333

# Path to existing projection
path_to_existing_raster = '/afs/ir/class/cs325b/gdal_tutorial/data/F182013.v4c_web.stable_lights.avg_vis.tif'

# generate arbitrary data to write to a new raster
grid = np.arange(num_x_pixels*num_y_pixels, dtype=np.float32).reshape(num_y_pixels, num_x_pixels)
grid = grid / np.max(grid)

################################################

# Create a GeoTIFF driver
driver = gdal.GetDriverByName('GTiff')

# Create a GDAL Dataset
dataset = driver.Create(
  output_path,            # path to output TIF file
  xsize=num_x_pixels,     # width of created raster in pixels
  ysize=num_y_pixels,     # height of created raster in pixels
  bands=num_bands,        # number of bands
  eType=raster_data_type  # raster data type
)

# Set the GeoTransform
# - The upper left corner of the upper left pixel in the created raster is at position (x_min_loc, y_min_loc) where x_min_loc and y_min_loc are in the units of the raster's projection. For example, if the projection is in units of degree lat/lon, then x_min_loc and y_min_loc correspond to the longitude and latitude (respectively) of the upper left pixel in the created raster
# - Likewise, PIXEL_WIDTH and PIXEL_HEIGHT indicate the resolution of each pixel in the created raster in the units of the projection. For example, if the projection uses units of meters, then PIXEL_WIDTH=30 and PIXEL_HEIGHT=30 would indicate that the raster has 30m/pixel resolution.
dataset.SetGeoTransform([
  x_min_loc,
  PIXEL_WIDTH,
  0,
  y_min_loc,
  0,
  -PIXEL_HEIGHT # due to how GDAL works, this must be negative
])

# Set the raster projection
# - Typically we use the projection of some existing raster
existing_raster_ds = gdal.Open(path_to_existing_raster)
existing_proj = existing_raster_ds.GetProjection()
dataset.SetProjection(existing_proj)

# Select a band to write data into
# - bands in GDAL are indexed starting from 1 to dataset.GetRasterCount()
# - grid is a 2-D numpy array of values that you want to write to this raster band
band = dataset.GetRasterBand(1)
band.WriteArray(grid)

# force GDAL to write the raster to disk
dataset.FlushCache()