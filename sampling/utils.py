import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from dfply import *
from gp.kernels import SquaredExponentialKernel


KERNEL = SquaredExponentialKernel(0.1, 0.05)
NOISE_LVL = 0.01
B = 1e-5 * np.eye(1001)
GP_MAX_SIZE = 5000


class SamplingLogger(object):
    """
    Logger to store results from predicting poverty with sampling.
    """
    def __init__(self, name):
        self.name = name
        self.r2s = []
        self.iterations = []
        self.idxs_matrix = []

    def tick(self, i, df, preds):
        self.r2s.append(calc_r2_score(df, preds))
        self.iterations.append(i)

    def clear_run(self, idxs):
        self.idxs_matrix.append(idxs)

    def save(self, data_subdir):
        self.iterations = np.array(self.iterations)
        self.r2s = np.array(self.r2s)
        self.idxs_matrix = np.array(self.idxs_matrix)
        filename = f"results/{data_subdir}/sampling_{self.name}.pkl"
        pickle.dump(self, open(filename, "wb"))


def calc_r2_score(df, preds, level="taluk"):
    """
    Calculate R2 score for a data frame and its predictions.
    """
    df["tmp_pred"] = preds
    if level == "village":
        return r2_score(df["true"], df.tmp_pred)
    elif level == "taluk":
        tmp = df >> group_by(X.taluk_idx) \
                 >> summarise(true = X.true.mean(), pred = X.tmp_pred.mean())
        return r2_score(tmp["true"], tmp["pred"])
    elif level == "district":
        tmp = df >> group_by(X.district_idx) \
                 >> summarise(true = X.true.mean(), pred = X.tmp_pred.mean())
        return r2_score(tmp["true"], tmp["pred"])
    else:
        raise ValueError

