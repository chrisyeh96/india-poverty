# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import shapefile
import gdal
import os
from multiprocessing import Pool
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from geotiling import GeoProps


def load_dataset():
  """
  Load the India dataframe.
  """
  india_df = pd.read_csv("../data/india.csv")
  return india_df

def load_disk_files():
  """
  Return a set of all imagery files presently on disk.
  """
  return set([s.strip() for s in open("../data/disk_bucket.txt", "r").readlines()])

def load_viirs_data(india_df, diameter=14):
  """
  Load numpy array continaing VIIRS night-lights
  """
  lights_tif = "../data/bucket/SVDNB_npp_20150101-20151231_75N060E_vcm-orm-ntl_v10_c201701311200.avg_rade9.tif"
  viirs_tif = gdal.Open(lights_tif)
  viirs_props = GeoProps()
  viirs_props.import_geogdal(viirs_tif)
  viirs_night_intensities = np.zeros(len(india_df))

  for i, idx in tqdm(enumerate(india_df.index)):
    lat = india_df["latitude"][idx]
    lng = india_df["longitude"][idx]
    x, y = viirs_props.lonlat2colrow(lng, lat)
    patch = viirs_tif.ReadAsArray(x - diameter // 2, y - diameter // 2, diameter, diameter)
    if patch is None:
      pass
    else:
      viirs_night_intensities[i] = np.mean(patch)

  return viirs_night_intensities


def load_dmsp_data(india_df, diameter=7):
  """
  Load numpy array containing DMSP night-lights.
  """
  lights_tif = "../data/bucket/F182013.v4c_web.stable_lights.avg_vis.tif"
  dmsp_tif = gdal.Open(lights_tif)
  dmsp_props = GeoProps()
  dmsp_props.import_geogdal(dmsp_tif)
  dmsp_night_intensities = np.zeros(len(india_df))

  for i, idx in tqdm(enumerate(india_df.index)):
    lat = india_df["latitude"][idx]
    lng = india_df["longitude"][idx]
    x, y = dmsp_props.lonlat2colrow(lng, lat)
    patch = dmsp_tif.ReadAsArray(x - diameter // 2, y - diameter // 2, diameter, diameter)
    if patch is None:
      pass
    else:
      dmsp_night_intensities[i] = np.mean(patch)

  return dmsp_night_intensities


def load_state_data(india_df):
  """
  Load data for which state each village belongs to
  """
  global _get_data_for_idx

  state_shapes = shapefile.Reader("../data/india_shape_files/IND_adm1").shapes()
  state_shape_points = [shape.points for shape in state_shapes]
  state_polygons = [Polygon(p) for p in state_shape_points]
  district_shapes = shapefile.Reader("../data/india_shape_files/IND_adm2").shapes()
  district_shape_points = [shape.points for shape in district_shapes]
  district_polygons = [Polygon(p) for p in district_shape_points]

  geography_df = pd.read_csv("../data/india_shape_files/IND_adm2.csv")
  state_names = np.zeros(len(india_df), dtype=str)
  district_names = np.zeros(len(india_df), dtype=str)

  def _get_data_for_idx(i):
    row = india_df.iloc[i,:]
    point = Point(row["longitude"], row["latitude"])
    contains = [p.contains(point) for p in state_polygons]
    state_idx = np.argmax(contains) if np.any(contains) else -1
    contains = [p.contains(point) for p in district_polygons]
    district_idx = np.argmax(contains) if np.any(contains) else -1
    return state_idx, district_idx

  print("Loading village state data...")
  with Pool(os.cpu_count()) as pool:
    state_district_idxs = list(tqdm(pool.imap(_get_data_for_idx,
                                              range(len(india_df))),
                                    total=len(india_df)))
  state_idxs = np.array(state_district_idxs)[:,0]
  district_idxs = np.array(state_district_idxs)[:,1]
  state_names = np.array(geography_df["NAME_1"][state_idxs].reset_index().iloc[:,1])
  state_names[state_idxs < 0] = ""
  district_names = np.array(geography_df["NAME_2"][district_idxs].reset_index().iloc[:,1])
  district_names[district_idxs < 0] = ""

  return state_idxs, state_names, district_idxs, district_names

def z_transform(col):
  mu = np.mean(col)
  sd = np.std(col)
  print("mu, sd:", (mu, sd))
  return (col - mu) / sd


if __name__ == "__main__":

  print("Loading dataset and disk files...")
  india_df = load_dataset()
  disk_files = load_disk_files()

  # filter out anomalies
  india_df = india_df[india_df["secc_cons_per_cap_scaled"] > 0]

  # filter out missing lat/lngs
  india_df = india_df[np.logical_and(india_df["latitude"].notnull(),
                                     india_df["longitude"].notnull())]

  print("Filtering out missing files...")
  india_df["l8_path"] = india_df["id"].apply(lambda i: "l8_median_india_vis_500x500_{:.1f}.tif".format(i))
  india_df["s1_path"] = india_df["id"].apply(lambda i: "s1_median_india_vis_500x500_{:.1f}.tif".format(i))
  india_df = india_df[np.logical_and(india_df["l8_path"].isin(disk_files),
                                     india_df["s1_path"].isin(disk_files))]

  print("Gathering night-light data...")
  india_df["dmsp"] = load_dmsp_data(india_df)
  india_df["viirs"] = load_viirs_data(india_df)

  print("Gathering state data...")
  state_idxs, state_names, district_idxs, district_names = load_state_data(india_df)
  india_df["state_idx"] = state_idxs
  india_df["state_name"] = state_names
  india_df["district_idx"] = district_idxs
  india_df["district_name"] = district_names

  # filter out missing states
  india_df = india_df[india_df["state_idx"] >= 0]

  print("Saving to CSV...")
  india_df["secc_cons_per_cap_scaled"] = z_transform(india_df["secc_cons_per_cap_scaled"])
  india_df.to_csv("../data/india_processed.csv", index=False)

