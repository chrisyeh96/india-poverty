import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


home_dir = os.path.expanduser("~")

if __name__ == "__main__":

  arg_parser = argparse.ArgumentParser(description="parser for transfer-learning")

  arg_parser.add_argument("--name", type=str, default=None, help="name for the model")
  arg_parser.add_argument("--use-grouped-labels", action="store_true")

  args = arg_parser.parse_args()

  losses = {
    "train": np.load("{}/predicting-poverty/models/{}/losses_train.npy".format(home_dir, args.name)),
    "val": np.load("{}/predicting-poverty/models/{}/losses_val.npy".format(home_dir, args.name))
  }
  rsq = {
    "train": np.load("{}/predicting-poverty/models/{}/rsq_train.npy".format(home_dir, args.name)),
    "val": np.load("{}/predicting-poverty/models/{}/rsq_val.npy".format(home_dir, args.name))
  }

  y_pred = np.load("{}/predicting-poverty/models/{}/y_pred.npy".format(home_dir, args.name))
  y_valid = np.load("{}/predicting-poverty/models/{}/y_true.npy".format(home_dir, args.name))

  if args.use_grouped_labels:
    df = pd.DataFrame(np.array([y_valid, y_pred]).T)
    grouped_y_pred = df.groupby(0)[1].mean()
    metrics.r2_score(grouped_y_pred.index, grouped_y_pred)
    print("==========================================================================")
    print("== ", args.name)
    print("Best training R^2:", np.max(rsq["train"]))
    print("Best validation R^2:", metrics.r2_score(grouped_y_pred.index, grouped_y_pred))
  else:
    print("==========================================================================")
    print("== ", args.name)
    print("Best training R^2:", np.max(rsq["train"]))
    print("Best validation R^2:", np.max(rsq["val"]))

