import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
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
    arg_parser.add_argument("--label", type=str,
                            default="secc_cons_per_cap_scaled")
    arg_parser.add_argument("--frac", type=float, default=1.0)
    arg_parser.add_argument("--data-subdir", default=0, type=int)
    args = arg_parser.parse_args()

    print_hr("Loading datasets and running baseline models...")

    if args.data_subdir:
        folds = [args.data_subdir]
    else:
        folds = range(1, 30)

    for data_subdir in folds:

        india_train = pd.read_csv("../data/fold_%d/train.csv" % data_subdir)
        india_valid = pd.read_csv("../data/fold_%d/valid.csv" % data_subdir)
        india_test = pd.read_csv("../data/fold_%d/test.csv"% data_subdir)

        india_train = india_train.sample(frac=args.frac)
        india_valid = india_valid.sample(frac=args.frac)

        X_train = pd.concat([india_train["dmsp"],
                             india_train["viirs"],
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

        np.save("../results/fold_%d/y_pred_gbm.npy" % data_subdir, preds["gbm"])
        np.save("../results/fold_%d/y_pred_ridge.npy" % data_subdir,
                preds["ridge"])
