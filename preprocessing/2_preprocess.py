import numpy as np
import pandas as pd
import shapely.geometry as sgeom
import shapefile
import gdal
from dfply import *
from pathlib import Path
from multiprocessing import Pool
from shapely.geometry import Polygon, Point
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from geotiling import GeoProps
from geopy.distance import vincenty


home_dir = str(Path.home())


def load_dataset():
    """
    Load the India dataframe.
    """
    india_df = pd.read_csv("data/india_with_survey.csv")
    return india_df

def load_disk_files():
    """
    Return a set of all imagery files presently on disk.
    """
    return set([s.strip() for s in open("data/disk_bucket.txt", "r").readlines()])

def load_viirs_data(india_df, diameter=14):
    """
    Load numpy array continaing VIIRS night-lights
    """
    lights_tif = f"{home_dir}/nightlights/SVDNB_npp_20150101-20151231_75N060E_vcm-orm-ntl_v10_c201701311200.avg_rade9.tif"
    print("VIIRS", lights_tif)
    viirs_tif = gdal.Open(lights_tif)
    viirs_props = GeoProps()
    viirs_props.import_geogdal(viirs_tif)
    viirs_night_intensities = np.zeros(len(india_df))

    for i, idx in tqdm(enumerate(india_df.index), total=len(india_df.index)):
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
    lights_tif = "%s/nightlights/F182013.v4c_web.stable_lights.avg_vis.tif" % home_dir
    print("DMSP", lights_tif)
    dmsp_tif = gdal.Open(lights_tif)
    dmsp_props = GeoProps()
    dmsp_props.import_geogdal(dmsp_tif)
    dmsp_night_intensities = np.zeros(len(india_df))

    for i, idx in tqdm(enumerate(india_df.index), total=len(india_df.index)):
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

    state_shapes = shapefile.Reader("data/india_shape_files/IND_adm1").shapes()
    state_polygons = [sgeom.shape(s.__geo_interface__) for s in state_shapes]
    district_shapes = shapefile.Reader("data/india_shape_files/IND_adm2").shapes()
    district_polygons = [sgeom.shape(s.__geo_interface__) for s in district_shapes]
    taluk_shapes = shapefile.Reader("data/india_shape_files/IND_adm3").shapes()
    taluk_polygons = [sgeom.shape(s.__geo_interface__) for s in taluk_shapes]
    district_centroids = [(p.centroid.x, p.centroid.y) for i, p in enumerate(district_polygons)]
    clustering = AgglomerativeClustering(n_clusters=100)
    pred_clusters = clustering.fit_predict(district_centroids)
    district_idx_to_cluster_mapping = {i: c for i, c in enumerate(pred_clusters)}
    district_idx_to_cluster_mapping[-1] = -1

    states_df = pd.read_csv("data/india_shape_files/IND_adm1.csv")
    districts_df = pd.read_csv("data/india_shape_files/IND_adm2.csv")
    taluks_df = pd.read_csv("data/india_shape_files/IND_adm3.csv")
    state_names = np.zeros(len(india_df), dtype=str)
    district_names = np.zeros(len(india_df), dtype=str)
    taluk_names = np.zeros(len(india_df), dtype=str)

    def _get_data_for_idx(i):
        row = india_df.iloc[i,:]
        point = Point(row["longitude"], row["latitude"])
        contains = [p.contains(point) for p in state_polygons]
        state_idx = np.argmax(contains) if np.any(contains) else -1
        contains = [p.contains(point) for p in district_polygons]
        district_idx = np.argmax(contains) if np.any(contains) else -1
        contains = [p.contains(point) for p in taluk_polygons]
        taluk_idx = np.argmax(contains) if np.any(contains) else -1
        return state_idx, district_idx, taluk_idx

    print("Loading village state data...")
    with Pool(8) as pool:
        idxs = list(tqdm(pool.imap(_get_data_for_idx, range(len(india_df))),
                         total=len(india_df)))
    state_idxs = np.array(idxs)[:,0]
    district_idxs = np.array(idxs)[:,1]
    taluk_idxs = np.array(idxs)[:,2]
    cluster_idxs = np.array([district_idx_to_cluster_mapping[i] for i in district_idxs])
    state_names = np.array(states_df["NAME_1"][state_idxs].reset_index().iloc[:,1])
    state_names[state_idxs < 0] = ""
    state_idxs = np.array(states_df["ID_1"][state_idxs].reset_index().iloc[:,1])
    district_names = np.array(districts_df["NAME_2"][district_idxs].reset_index().iloc[:,1])
    district_names[district_idxs < 0] = ""
    district_idxs = np.array(districts_df["ID_2"][district_idxs].reset_index().iloc[:,1])
    taluk_names = np.array(taluks_df["NAME_3"][taluk_idxs].reset_index().iloc[:,1])
    taluk_names[taluk_idxs < 0] = ""
    taluk_idxs = np.array(taluks_df["ID_3"][taluk_idxs].reset_index().iloc[:,1])
    return state_idxs, state_names, district_idxs, district_names, taluk_idxs, taluk_names, cluster_idxs


if __name__ == "__main__":

    print("Loading dataset and disk files...")
    india_df = load_dataset()
    disk_files = load_disk_files()

    # filter out anomalies
    india_df >>= mask(X.secc_cons_per_cap_scaled > 0)

    # filter out missing lat/lngs
    india_df >>= mask(X.latitude.notnull() & X.longitude.notnull())

    print("Filtering out missing files...")
    india_df >>= mutate(l8_path = X.id.apply(lambda i: f"l8_median_india_vis_500x500_{i:.1f}.tif"),
                        s1_path = X.id.apply(lambda i: f"s1_median_india_vis_500x500_{i:.1f}.tif"))
    india_df >>= mask(X.l8_path.isin(disk_files) & X.s1_path.isin(disk_files))

    print("Gathering night-light data...")
    india_df["dmsp"] = load_dmsp_data(india_df)
    india_df["viirs"] = load_viirs_data(india_df)

    print("Gathering state data...")
    state_idxs, state_names, district_idxs, district_names, taluk_idxs, taluk_names, cluster_idxs = load_state_data(india_df)
    india_df["state_idx"] = state_idxs
    india_df["state_name"] = state_names
    india_df["district_idx"] = district_idxs
    india_df["district_name"] = district_names
    india_df["taluk_idx"] = taluk_idxs
    india_df["taluk_name"] = taluk_names
    india_df["cluster_idx"] = cluster_idxs

    # filter out missing states
    india_df >>= mask(X.state_name.notnull())

    print("Saving to CSV...")
    india_df.to_csv("data/india_processed.csv", index=False)
