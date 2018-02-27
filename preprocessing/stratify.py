# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from geopy.distance import vincenty


def load_processed_data():
  return pd.read_csv("../data/india_processed.csv")

def stratify_kfold(df, n_folds=5):
  """
  Stratify by district into various folds.
  """
  district_idxs = np.arange(np.max(df["district_idx"]))
  np.random.shuffle(district_idxs)
  folds = np.array_split(district_idxs, n_folds)
  result = []
  for fold in folds:
    test = df[df["district_idx"].isin(fold)]
    train = df[~df["district_idx"].isin(fold)]
    train = train.sample(frac=1.0)
    valid = train.iloc[:len(train)//5,:]
    train = train.iloc[len(train)//5:,:]
    result.append({
      "train": train,
      "valid": valid,
      "test": test
    })
  return result

def calc_distances_for_fold(fold):
  """
  Calculate distances, for each point in the test set, to the nearest point
  in the training set.
  """
  train = pd.concat([fold["train"], fold["valid"]])
  test = fold["test"]
  coords_train = np.array([train["latitude"], train["longitude"]]).T
  dists_test = []

  for i in tqdm(range(len(test))):
    row = test.iloc[i,:]
    coords = np.array([row["latitude"], row["longitude"]]).T
    dists = np.linalg.norm(coords_train - np.expand_dims(coords, 0), axis=1)
    j = np.argmin(dists)
    nearest = coords_train[j,:]
    dist = vincenty(coords, nearest).meters
    dists_test.append(dist)

  return np.array(dists_test)

if __name__ == "__main__":

  print("Loading and stratifying dataset...")
  india_df = load_processed_data()
  folds = stratify_kfold(india_df)

  for i, fold in enumerate(folds):
    print("Processing fold {}...".format(i+1))
    base_dir = "../data/fold_{}".format(i+1)
    os.mkdir(base_dir)
    np.save(os.path.join(base_dir, "dists.npy"), calc_distances_for_fold(fold))
    fold["train"].to_csv(os.path.join(base_dir, "train.csv"))
    fold["valid"].to_csv(os.path.join(base_dir, "valid.csv"))
    fold["test"].to_csv(os.path.join(base_dir, "test.csv"))
