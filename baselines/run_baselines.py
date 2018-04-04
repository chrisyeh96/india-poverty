import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from skimage.feature import hog, daisy
from skimage import data, color, exposure
from sklearn import metrics


def print_hr(s=""):
  print("=================================================================")
  print("== ", s)


def run_nightlight_models(X_train, X_valid, y_train, y_valid, name, save):

  gbm = GradientBoostingRegressor(loss="ls", n_estimators=25, max_depth=10)
  gbm.fit(X_train, y_train)
  print_hr("GBM")
  print("R^2:", gbm.score(X_valid, y_valid))

  train_scores, valid_scores = [], []
  for alpha in np.logspace(-5, 3, num=20):
    lm = Ridge(alpha=alpha, normalize=True)
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_train)
    train_scores.append(metrics.r2_score(y_train, y_pred))
    y_pred = lm.predict(X_valid)
    valid_scores.append(metrics.r2_score(y_valid, y_pred))
  print_hr("Ridge")
  print("R^2:", np.max(valid_scores))

  if save:
    os.mkdir("../models/{}_gbm".format(name))
    y_pred = gbm.predict(X_valid)
    np.save("../models/{}_gbm/y_pred.npy".format(name), y_pred)
    np.save("../models/{}_gbm/y_true.npy".format(name), y_valid)


def run_knn_model(X_train, X_valid, y_train, y_valid):

  gp = KNeighborsRegressor(n_neighbors=30)
  gp.fit(X_train, y_train)
  print_hr("KNN")
  print("R^2:", gp.score(X_valid, y_valid))
  print("R^2 no outliers:", gp.score(X_valid[y_valid < 3], y_valid[y_valid < 3]))


if __name__ == "__main__":

  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--label", type=str, default="secc_cons_per_cap_scaled")
  arg_parser.add_argument("--name", type=str)
  arg_parser.add_argument("--data-subdir", type=str)
  arg_parser.add_argument("--frac", type=float, default=1.0)
  arg_parser.add_argument("--save", action="store_true")
  args = arg_parser.parse_args()

  print_hr("Loading datasets...")

  india = pd.read_csv("../data/india_processed.csv")
  india_train = pd.read_csv("../data/{}/train.csv".format(args.data_subdir))
  india_valid = pd.read_csv("../data/{}/test.csv".format(args.data_subdir))
  # india_test = pd.read_csv("../data/{}/test.csv".format(args.data_subdir))

  # print_hr("Correlation India | VIIRS")
  # print("r:", india["viirs"].corr(india[args.label]))

  # print_hr("Correlation India | DMSP")
  # print("r:", india["dmsp"].corr(india[args.label]))

  india_train = india_train.sample(frac=args.frac)

  X_train = pd.concat([india_train["dmsp"], india_train["viirs"],
                       np.log(india_train["dmsp"] + 1),
                       np.log(india_train["viirs"] + 1)], axis=1)
  X_valid = pd.concat([india_valid["dmsp"], india_valid["viirs"],
                       np.log(india_valid["dmsp"] + 1),
                       np.log(india_valid["viirs"] + 1)], axis=1)
  y_train = india_train[args.label]
  y_valid = india_valid[args.label]

  run_nightlight_models(X_train, X_valid, y_train, y_valid,
                        name=args.name, save=args.save)

  X_train = pd.concat([india_train["latitude"], india_train["longitude"]], axis=1)
  X_valid = pd.concat([india_valid["latitude"], india_valid["longitude"]], axis=1)
  y_train = india_train[args.label]
  y_valid = india_valid[args.label]

  run_knn_model(X_train, X_valid, y_train, y_valid)
