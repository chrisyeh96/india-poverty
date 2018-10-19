import numpy as np
import scipy
import pandas as pd
import matplotlib as mpl
from argparse import ArgumentParser
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.stats import pearsonr
from tqdm import tqdm
from utils import CoeffSamplingLogger, anisotropic_kernel, get_taluk_df, \
                  MAX_N_SAMPLES

np.random.seed(123)
np.warnings.filterwarnings("ignore")


def sample_greedy(df, n_reps=5, n_gp_samples=10):
  print("== Sampling for coefficient estimation, greedily...")
  logger = CoeffSamplingLogger("greedy")
  for _ in range(n_reps):
    n_samples = min(len(df) - 1, MAX_N_SAMPLES)
    idxs = [np.random.randint(len(df))]
    for i in tqdm(range(1, n_samples)):
      df_sampled = df.iloc[df.index.isin(idxs),:]
      df_rest = df.iloc[~df.index.isin(idxs),:]
      logger.tick(df_sampled)
      gp = GaussianProcessRegressor(kernel=anisotropic_kernel, normalize_y=True)
      gp.fit(df_sampled.loc[:,("lat", "lng", "pred")], df_sampled["true"] - df_sampled["pred"])
      preds = gp.sample_y(df_rest.loc[:, ("lat", "lng", "pred")], n_samples = n_gp_samples) + \
              df_rest["pred"][:, np.newaxis]
      betas = np.zeros_like(preds)
      for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
          elec = np.r_[df_sampled["electrification"], df_rest.iloc[i,:]["electrification"]]
          cons = np.r_[df_sampled["true"], preds[i,j]]
          betas[i,j] = pearsonr(elec, cons)[0]
      idx = df_rest.index[np.argmax(np.std(betas, axis=1))]
      idxs.append(idx)
    logger.clear_run()
  return logger


def sample_random(df, n_reps = 10):
  print("== Sampling for coefficient estimation, randomly...")
  logger = CoeffSamplingLogger("random")
  for _ in range(n_reps):
    idxs_random = np.arange(len(df))
    np.random.shuffle(idxs_random)
    n_samples = min(len(df) - 1, MAX_N_SAMPLES)
    for i in tqdm(range(1, n_samples)):
      df_sampled = df.iloc[idxs_random[:i],:]
      logger.tick(df_sampled)
    logger.clear_run()
  return logger


if __name__ == "__main__":

  arg_parser = ArgumentParser()
  arg_parser.add_argument("--fold_idx", type=str, default=None)
  args = arg_parser.parse_args()
  fold_idx = args.fold_idx

  df_train = pd.read_csv("../results/fold_%s/sampling_train.csv" % fold_idx)
  df_val = pd.read_csv("../results/fold_%s/sampling_val.csv" % fold_idx)
  df = pd.concat([df_train, df_val]).reset_index()
  print("Shape of aggregated test set:", df.shape)

  logs = sample_greedy(df)
  logs.save(fold_idx)
  logs = sample_random(df)
  logs.save(fold_idx)