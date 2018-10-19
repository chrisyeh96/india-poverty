import numpy as np
import scipy
import pandas as pd
import matplotlib as mpl
import pickle
from argparse import ArgumentParser
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.linalg import cholesky
from scipy.linalg import solve_triangular
from tqdm import tqdm
from utils import anisotropic_kernel, isotropic_kernel, SamplingLogger, \
                  get_taluk_df, MAX_N_SAMPLES


np.random.seed(123)
np.warnings.filterwarnings('ignore')



def greedy_selection(init_kernel, n_samples, X, y, chol_inv=False):
  """
  Note that using the Cholesky decomposition here is buggy.
  """
  remaining_indices = set(np.arange(len(X)))
  picked_indices = set()

  first_pick = np.random.randint(0, len(X))
  picked_indices.add(first_pick)
  remaining_indices.remove(first_pick)

  greedy_order = [first_pick]

  for i in tqdm(range(n_samples - 1)):

    gp = GaussianProcessRegressor(kernel=init_kernel, normalize_y=True)

    # Fit the GP with the points we've sampled so far
    # this does likelihood maximization to find kernel hyperparameters
    # in addition to doing GP regression
    gp.fit(X[np.array(sorted(list(picked_indices)))], y[np.array(sorted(list(picked_indices)))])

    # This is kernel with ML estimates for hyperparams
    kernel = gp.kernel_

    # Compute the variance at the points that are available to sample
    var_y = np.diag(kernel(X[np.array(sorted(list(remaining_indices))),:]))

    # Find the covariance matrix for the sampled points and invert
    cov_AA = kernel(X[np.array(sorted(list(picked_indices)))])
    if chol_inv:
      chol = cholesky(cov_AA)
      inv_chol = scipy.linalg.solve_triangular(chol, np.identity(chol.shape[0]))
      inv_cov_AA = np.dot(inv_chol,np.transpose(inv_chol))
    else:
      inv_cov_AA = np.linalg.inv(cov_AA)

    # Find the kernel matrix of the picked points wrt remaining points
    cov_Ay = kernel(X[np.array(sorted(list(picked_indices)))], X[np.array(sorted(list(remaining_indices)))])
    cov_yA = np.transpose(cov_Ay)

    # Run the selection procedure over the remaining indices
    # (note it is possible to speed this up: there is a section in the
    # paper on speeding up this process for large numbers of points)
    max_delta_j, greedy_idx = float("-inf"), -1

    for idx, j in enumerate(sorted(list(remaining_indices))):

      try:
        cov_AbarAbar = kernel(X[np.array(sorted(list(remaining_indices - set([j])))),:])
      except:
        import pdb
        pdb.set_trace()
      if chol_inv:
        chol = np.linalg.cholesky(cov_AbarAbar)
        inv_chol = scipy.linalg.solve_triangular(chol, np.identity(chol.shape[0]))
        inv_cov_AbarAbar = np.dot(inv_chol,np.transpose(inv_chol))
      else:
        inv_cov_AbarAbar = np.linalg.inv(cov_AbarAbar)

      cov_Abary = kernel(X[np.array(sorted(list(remaining_indices - set([j]))))], X[j,:][np.newaxis,:])
      cov_yAbar = np.transpose(cov_Abary)
      delta_j = (var_y[idx] - np.dot(np.dot(cov_yA[idx,:][np.newaxis,:],inv_cov_AA),
                                     cov_Ay[:,idx][:,np.newaxis])) / (var_y[idx] - np.dot(np.dot(cov_yAbar, inv_cov_AbarAbar), cov_Abary))
      delta_j = delta_j.flatten()[0]

      if delta_j > max_delta_j:
        max_delta_j = delta_j
        greedy_idx = j

    picked_indices.add(greedy_idx)
    remaining_indices.remove(greedy_idx)
    greedy_order.append(greedy_idx)

  return greedy_order

def sample_greedy_with_sat(df_train, df_val, n_reps=5):
  """
  Sample greedily, building on top of satellite predictions.
  """
  print("== Sampling greedily with satellite predictions...")
  logger = SamplingLogger("greedy_sat")
  for _ in range(n_reps):
    n_samples = min(len(df_train) - 1, MAX_N_SAMPLES)
    idxs = greedy_selection(
      anisotropic_kernel, n_samples,
      df_train.loc[:,("lat", "lng", "pred")].values,
      np.array(df_train["true"] - df_train["pred"]), chol_inv=False)
    for i in tqdm(range(1, len(idxs))):
      df_sampled = df_train.iloc[idxs[:i],:]
      gp = GaussianProcessRegressor(kernel=isotropic_kernel, normalize_y=True)
      gp.fit(df_sampled.loc[:,("lat", "lng")], df_sampled["true"] - df_sampled["pred"])
      preds = gp.predict(df_val.loc[:,("lat", "lng")]) + df_val["pred"]
      logger.tick(df_val, preds)
    logger.clear_run()
  return logger

def sample_greedy_no_sat(df_train, df_val, n_reps=5):
  """
  Sample greedily, without building on satellite predictions.
  """
  print("== Sampling greedily without satellite predictions...")
  logger = SamplingLogger("greedy_nosat")
  for _ in range(n_reps):
    n_samples = min(len(df_train) - 1, MAX_N_SAMPLES)
    idxs = greedy_selection(
      isotropic_kernel, n_samples,
      df_train.loc[:,("lat", "lng")].values,
      np.array(df_train["true"]), chol_inv=False)
    for i in tqdm(range(1, len(idxs))):
      df_sampled = df_train.iloc[idxs[:i],:]
      gp = GaussianProcessRegressor(kernel=isotropic_kernel, normalize_y=True)
      gp.fit(df_sampled.loc[:,("lat", "lng")], df_sampled["true"])
      preds = gp.predict(df_val.loc[:,("lat", "lng")])
      logger.tick(df_val, preds)
    logger.clear_run()
  return logger

def sample_random_with_sat(df_train, df_val, n_reps=10):
  """
  Randomly sample, building on top of satellite predictions.
  """
  print("== Sampling randomly with satellite predictions...")
  logger = SamplingLogger("random_sat")
  for _ in range(n_reps):
    idxs_random = np.arange(len(df_train))
    np.random.shuffle(idxs_random)
    n_samples = min(len(df_train) - 1, MAX_N_SAMPLES)
    for i in tqdm(range(1, n_samples)):
      df_sampled = df_train.iloc[idxs_random[:i],:]
      gp = GaussianProcessRegressor(kernel=isotropic_kernel, normalize_y=True)
      gp.fit(df_sampled.loc[:,("lat", "lng")], df_sampled["true"] - df_sampled["pred"])
      preds = df_val["pred"] + gp.predict(df_val.loc[:,("lat", "lng")])
      logger.tick(df_val, preds)
    logger.clear_run()
  return logger

def sample_random_no_sat(df_train, df_val, n_reps=10):
  """
  Randomly sample, without building on top of satellite predictions.
  """
  print("== Sampling randomly without satellite predictions...")
  logger = SamplingLogger("random_nosat")
  for _ in range(n_reps):
    idxs_random = np.arange(len(df_train))
    np.random.shuffle(idxs_random)
    n_samples = min(len(df_train) - 1, MAX_N_SAMPLES)
    for i in tqdm(range(1, n_samples)):
      df_sampled = df_train.iloc[idxs_random[:i],:]
      gp = GaussianProcessRegressor(kernel=isotropic_kernel, normalize_y=True)
      gp.fit(df_sampled.loc[:,("lat", "lng")], df_sampled["true"])
      preds = gp.predict(df_val.loc[:,("lat", "lng")])
      logger.tick(df_val, preds)
    logger.clear_run()
  return logger


if __name__ == "__main__":

  arg_parser = ArgumentParser()
  arg_parser.add_argument("--fold_idx", type=str, default=None)
  args = arg_parser.parse_args()
  fold_idx = args.fold_idx

  electrification_data = pd.read_csv("../data/electrification.csv")
  df = pd.read_csv("../results/fold_%s/test_results.csv" % fold_idx)
  print("Length of original test set: %d" % len(df))

  df = df.merge(electrification_data, how="inner", left_on="id", right_on="village_id")
  df = df.loc[:,("smoothed", "true", "lat", "lng", "taluk_idx",
                 "district_idx", "state_idx", "pop", "electrification")]
  df = df.rename(columns={"smoothed": "pred"})

  print("Length of merged test set: %d" % len(df))
  df.to_csv("../results/fold_%s/sampling_village_level.csv" % fold_idx)

  df = get_taluk_df(df)
  print("Shape of aggregated test set:", df.shape)

  idxs = np.arange(len(df))
  train_idxs = np.random.choice(idxs, int(0.5 * len(idxs)), replace=False)
  val_idxs = np.array(list(set(idxs) - set(train_idxs)))
  df_train = df.iloc[train_idxs,:]
  df_val = df.iloc[val_idxs,:]
  df_train.to_csv("../results/fold_%s/sampling_train.csv" % fold_idx)
  df_val.to_csv("../results/fold_%s/sampling_val.csv" % fold_idx)

  logs = sample_random_with_sat(df_train, df_val)
  logs.save(fold_idx)
  logs = sample_random_no_sat(df_train, df_val)
  logs.save(fold_idx)
  logs = sample_greedy_with_sat(df_train, df_val)
  logs.save(fold_idx)
  logs = sample_greedy_no_sat(df_train, df_val)
  logs.save(fold_idx)
