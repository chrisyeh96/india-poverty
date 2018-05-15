import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from skimage.feature import hog, daisy
from skimage import data, color, exposure
from sklearn.metrics import r2_score


def print_hr(s=""):
  print("=================================================================")
  print("== ", s)


def run_nightlight_models(X_train, X_valid, X_test, y_train, y_valid, y_test):

  gbm = GradientBoostingRegressor(loss="ls", n_estimators=25, max_depth=10)
  gbm.fit(np.r_[X_train, X_valid], np.r_[y_train, y_valid])

  train_scores, valid_scores = [], []
  for alpha in np.logspace(-5, 3, num=10):
    lm = Ridge(alpha=alpha, normalize=True)
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_train)
    train_scores.append(r2_score(y_train, y_pred))
    y_pred = lm.predict(X_valid)
    valid_scores.append(r2_score(y_valid, y_pred))

  alpha = np.logspace(-5, 3, num=10)[np.argmax(valid_scores)]
  lm = Ridge(alpha=alpha, normalize=True)
  lm.fit(np.r_[X_train, X_valid], np.r_[y_train, y_valid])

  return {
    "gbm": gbm.predict(X_test),
    "ridge": lm.predict(X_test)
  }


def run_knn_model(X_train, X_valid, X_test, y_train, y_valid, y_test):

  knn = KNeighborsRegressor(n_neighbors=30)
  knn.fit(np.r_[X_train, X_valid], np.r_[y_train, y_valid])
  return knn.predict(X_test)


if __name__ == "__main__":

  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--label", type=str, default="secc_cons_per_cap_scaled")
  arg_parser.add_argument("--name", type=str)
  arg_parser.add_argument("--frac", type=float, default=1.0)
  arg_parser.add_argument("--save", action="store_true")
  args = arg_parser.parse_args()

  print_hr("Loading datasets and running baseline models...")

  fold_results = []

  for data_subdir in tqdm(range(1, 5 + 1)):

    india_train = pd.read_csv("../data/fold_{}/train.csv".format(data_subdir))
    india_valid = pd.read_csv("../data/fold_{}/valid.csv".format(data_subdir))
    india_test = pd.read_csv("../data/fold_{}/test.csv".format(data_subdir))

    india_train = india_train.sample(frac=args.frac)
    india_valid = india_valid.sample(frac=args.frac)

    X_train = pd.concat([india_train["dmsp"], india_train["viirs"],
                         np.log(india_train["dmsp"] + 1),
                         np.log(india_train["viirs"] + 1)], axis=1)
    X_valid = pd.concat([india_valid["dmsp"], india_valid["viirs"],
                         np.log(india_valid["dmsp"] + 1),
                         np.log(india_valid["viirs"] + 1)], axis=1)
    X_test = pd.concat([india_test["dmsp"], india_test["viirs"],
                         np.log(india_test["dmsp"] + 1),
                         np.log(india_test["viirs"] + 1)], axis=1)
    y_train = india_train[args.label]
    y_valid = india_valid[args.label]
    y_test = india_test[args.label]

    preds = run_nightlight_models(X_train, X_valid, X_test,
                                  y_train, y_valid, y_test)

    X_train = pd.concat([india_train["latitude"], india_train["longitude"]], axis=1)
    X_valid = pd.concat([india_valid["latitude"], india_valid["longitude"]], axis=1)
    X_test = pd.concat([india_test["latitude"], india_test["longitude"]], axis=1)
    preds["knn"] = run_knn_model(X_train, X_valid, X_test,
                                  y_train, y_valid, y_test)
    preds["true"] = y_test
    preds["district_idx"] = india_test["district_idx"]
    preds["taluk_idx"] = india_test["taluk_idx"]
    preds["fold"] = data_subdir
    fold_results.append(pd.DataFrame(preds))

  test_df = pd.concat(fold_results)
  print_hr("KNN")
  print("R2: %.3f" % r2_score(test_df["true"], test_df["knn"]))
  print("R2: %.3f" % r2_score(test_df.groupby("district_idx")["true"].mean(),
                              test_df.groupby("district_idx")["knn"].mean()))
  print("R2: %.3f" % r2_score(test_df.groupby("taluk_idx")["true"].mean(),
                              test_df.groupby("taluk_idx")["knn"].mean()))
  print_hr("Ridge")
  print("R2: %.3f" % r2_score(test_df["true"], test_df["ridge"]))
  print("R2: %.3f" % r2_score(test_df.groupby("district_idx")["true"].mean(),
                              test_df.groupby("district_idx")["ridge"].mean()))
  print("R2: %.3f" % r2_score(test_df.groupby("taluk_idx")["true"].mean(),
                              test_df.groupby("taluk_idx")["ridge"].mean()))
  print_hr("GBM")
  print("R2: %.3f" % r2_score(test_df["true"], test_df["gbm"]))
  print("R2: %.3f" % r2_score(test_df.groupby("district_idx")["true"].mean(),
                              test_df.groupby("district_idx")["gbm"].mean()))
  print("R2: %.3f" % r2_score(test_df.groupby("taluk_idx")["true"].mean(),
                              test_df.groupby("taluk_idx")["gbm"].mean()))
  print_hr("GBM no outliers")
  print("R2: %.3f" % r2_score(test_df["true"][test_df["true"] < 3],
                              test_df["knn"][test_df["true"] < 3]))


  if args.label == "secc_pov_rate":
    test_df.to_csv("./baseline_results_pov_rate.csv")
  else:
    test_df.to_csv("./baseline_results.csv")
