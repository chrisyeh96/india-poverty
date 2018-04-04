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
  cluster_idxs = np.arange(np.max(df["cluster_idx"]))
  np.random.shuffle(cluster_idxs)
  folds = np.array_split(cluster_idxs, n_folds)
  result = []

  for fold in folds:

    test = df[df["cluster_idx"].isin(fold)]
    train_val = df[~df["cluster_idx"].isin(fold)]
    train_val_districts = pd.unique(train_val["cluster_idx"])
    np.random.shuffle(train_val_districts)
    train_districts = train_val_districts[len(train_val_districts)//5:]
    valid_districts = train_val_districts[:len(train_val_districts)//5]
    train = train_val[train_val["cluster_idx"].isin(train_districts)]
    valid = train_val[train_val["cluster_idx"].isin(valid_districts)]
    result.append({
      "train": train,
      "valid": valid,
      "test": test
    })
  return result

def filter_nearby_villages(fold, margin):
  """
  Filter out villages to prevent overlap between training/testing villages.
  """
  coords_test = np.array([fold["test"]["latitude"],
                          fold["test"]["longitude"]]).T
  keep_train = np.ones(len(fold["train"]), dtype=bool)
  keep_valid = np.ones(len(fold["valid"]), dtype=bool)

  for i in tqdm(range(len(fold["train"]))):
    row = fold["train"].iloc[i,:]
    coords = np.array([row["latitude"], row["longitude"]]).T
    dists = np.linalg.norm(coords_test - np.expand_dims(coords, 0), axis=1)
    j = np.argmin(dists)
    nearest = coords_test[j,:]
    dist = vincenty(coords, nearest).meters
    if dist < margin:
      keep_train[i] = 0

  for i in tqdm(range(len(fold["valid"]))):
    row = fold["valid"].iloc[i,:]
    coords = np.array([row["latitude"], row["longitude"]]).T
    dists = np.linalg.norm(coords_test - np.expand_dims(coords, 0), axis=1)
    j = np.argmin(dists)
    nearest = coords_test[j,:]
    dist = vincenty(coords, nearest).meters
    if dist < margin:
      keep_valid[i] = 0

  return keep_train, keep_valid

if __name__ == "__main__":

  print("Loading and stratifying dataset...")
  margin = 2 * np.sqrt(2) * 1500

  india_df = load_processed_data()
  folds = stratify_kfold(india_df)

  for i, fold in enumerate(folds):
    print("Processing fold {}...".format(i+1))
    base_dir = "../data/fold_{}".format(i+1)
    os.mkdir(base_dir)
    keep_train, keep_valid = filter_nearby_villages(fold, margin)
    fold["train"] = fold["train"][keep_train]
    fold["valid"] = fold["valid"][keep_valid]
    print("Train below margin:", 1 - np.mean(keep_train))
    print("Valid below margin:", 1 - np.mean(keep_valid))
    np.save(os.path.join(base_dir, "keep_train.npy"), keep_train)
    np.save(os.path.join(base_dir, "keep_valid.npy"), keep_valid)
    label = pd.concat([fold["train"], fold["valid"]])["secc_cons_per_cap_scaled"]
    mu, std = np.mean(label), np.std(label)
    np.save(os.path.join(base_dir, "mu.npy"), mu)
    np.save(os.path.join(base_dir, "std.npy"), std)
    for k, v in fold.items():
      print(k, len(v))
      v["secc_cons_per_cap_scaled"] = (v["secc_cons_per_cap_scaled"] - mu) / std
      v.to_csv(os.path.join(base_dir, "{}.csv".format(k)))
