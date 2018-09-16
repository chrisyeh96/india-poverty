import numpy as np
import scipy
import pandas as pd
import matplotlib as mpl
import pickle
from argparse import ArgumentParser
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.linalg import cholesky
from scipy.linalg import solve_triangular
from tqdm import tqdm

np.random.seed(123)


isotropic_kernel = WhiteKernel() + \
                   ConstantKernel() * \
                   RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1000)) + \
                   RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1000))

anisotropic_kernel = WhiteKernel() + \
                     ConstantKernel() * \
                     RBF(length_scale=np.ones(3), length_scale_bounds=(1e-3, 1000)) + \
                     RBF(length_scale=np.ones(3), length_scale_bounds=(1e-3, 1000))


class SamplingLogger(object):

  def __init__(self, name):
    self.name = name
    self.r2s_matrix = []
    self.mses_matrix = []
    self.r2s = []
    self.mses = []

  def tick(self, df_val, preds):
    self.r2s.append(r2_score(df_val["true"], preds))
    self.mses.append(mean_squared_error(df_val["true"], preds))

  def clear_run(self):
    if self.r2s:
      self.r2s_matrix.append(self.r2s)
    if self.mses:
      self.mses_matrix.append(self.mses)
    self.r2s = []
    self.mses = []

  def save(self, fold_idx):
    pickle.dump(self, open("../results/fold_%s/sampling_%s.pkl" % (fold_idx, self.name), "wb"))


class CoeffSamplingLogger(object):

  def __init__(self, name):
    self.name = name
    self.coeffs_matrix = []
    self.coeffs_matrix_only_sampled = []
    self.coeffs = []
    self.coeffs_only_sampled = []

  def tick(self, df_sampled, df_rest, preds):
    preds = np.r_[preds, df_sampled["true"]]
    elec = np.r_[df_rest["electrification"], df_sampled["electrification"]]
    self.coeffs.append(np.polyfit(preds, elec, deg=1)[0])
    self.coeffs_only_sampled.append(np.polyfit(df_sampled["true"], df_sampled["electrification"], deg=1)[0])

  def clear_run(self):
    if self.coeffs:
      self.coeffs_matrix.append(self.coeffs)
    if self.coeffs_only_sampled:
      self.coeffs_matrix_only_sampled.append(self.coeffs_only_sampled)
    self.coeffs = []
    self.coeffs_only_sampled = []

  def save(self, fold_idx):
    pickle.dump(self, open("../results/fold_%s/coeff_sampling_%s.pkl" % (fold_idx, self.name), "wb"))


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

      cov_AbarAbar = kernel(X[np.array(sorted(list(remaining_indices - set([j])))),:])
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

def sample_greedy_with_sat(df_train, df_val, n_reps=3):
  """
  Sample greedily, building on top of satellite predictions.
  """
  logger = SamplingLogger("greedy_sat")
  coeff_logger = CoeffSamplingLogger("greedy_sat")
  for _ in range(n_reps):
    idxs = greedy_selection(
      anisotropic_kernel, len(df_train) - 1,
      df_train.loc[:,("lat", "lng", "pred")].as_matrix(),
      np.array(df_train["true"] - df_train["pred"]), chol_inv=False)
    for i in tqdm(range(1, len(idxs))):
      df_sampled = df_train.iloc[idxs[:i],:]
      gp = GaussianProcessRegressor(kernel=isotropic_kernel, normalize_y=True)
      gp.fit(df_sampled.loc[:,("lat", "lng")], df_sampled["true"] - df_sampled["pred"])
      preds = gp.predict(df_val.loc[:,("lat", "lng")]) + df_val["pred"]
      logger.tick(df_val, preds)
    logger.clear_run()
  df_train = pd.concat([df_train, df_val], axis=0)
  for _ in range(n_reps):
    idxs = greedy_selection(
      anisotropic_kernel, len(df_train) - 1,
      df_train.loc[:,("lat", "lng", "pred")].as_matrix(),
      np.array(df_train["true"] - df_train["pred"]), chol_inv=False)
    for i in tqdm(range(1, len(df_train) - 1)):
      df_sampled = df_train.iloc[idxs[:i],:]
      df_rest = df_train.iloc[np.setdiff1d(idxs, idxs[:i]),:]
      gp = GaussianProcessRegressor(kernel=isotropic_kernel, normalize_y=True)
      gp.fit(df_sampled.loc[:,("lat", "lng")], df_sampled["true"] - df_sampled["pred"])
      preds = df_rest["pred"] + gp.predict(df_rest.loc[:,("lat", "lng")])
      coeff_logger.tick(df_sampled, df_rest, preds)
    coeff_logger.clear_run()
  return logger, coeff_logger

def sample_greedy_no_sat(df_train, df_val, n_reps=3):
  """
  Sample greedily, without building on satellite predictions.
  """
  logger = SamplingLogger("greedy_nosat")
  coeff_logger = CoeffSamplingLogger("greedy_nosat")
  for _ in range(n_reps):
    idxs = greedy_selection(
      isotropic_kernel, len(df_train) - 1,
      df_train.loc[:,("lat", "lng")].as_matrix(),
      np.array(df_train["true"]), chol_inv=False)
    for i in tqdm(range(1, len(idxs))):
      df_sampled = df_train.iloc[idxs[:i],:]
      gp = GaussianProcessRegressor(kernel=isotropic_kernel, normalize_y=True)
      gp.fit(df_sampled.loc[:,("lat", "lng")], df_sampled["true"])
      preds = gp.predict(df_val.loc[:,("lat", "lng")])
      logger.tick(df_val, preds)
    logger.clear_run()
  df_train = pd.concat([df_train, df_val], axis=0)
  for _ in range(n_reps):
    idxs = greedy_selection(
      isotropic_kernel, len(df_train) - 1,
      df_train.loc[:,("lat", "lng")].as_matrix(),
      np.array(df_train["true"]), chol_inv=False)
    for i in tqdm(range(1, len(df_train) - 1)):
      df_sampled = df_train.iloc[idxs[:i],:]
      df_rest = df_train.iloc[np.setdiff1d(idxs, idxs[:i]),:]
      gp = GaussianProcessRegressor(kernel=isotropic_kernel, normalize_y=True)
      gp.fit(df_sampled.loc[:,("lat", "lng")], df_sampled["true"])
      preds = gp.predict(df_rest.loc[:,("lat", "lng")])
      coeff_logger.tick(df_sampled, df_rest, preds)
    coeff_logger.clear_run()
  return logger, coeff_logger

def sample_random_with_sat(df_train, df_val, n_reps=10):
  """
  Randomly sample, building on top of satellite predictions.
  """
  logger = SamplingLogger("random_sat")
  coeff_logger = CoeffSamplingLogger("random_sat")
  for _ in range(n_reps):
    idxs_random = np.arange(len(df_train))
    np.random.shuffle(idxs_random)
    for i in tqdm(range(1, len(df_train))):
      df_sampled = df_train.iloc[idxs_random[:i],:]
      gp = GaussianProcessRegressor(kernel=isotropic_kernel, normalize_y=True)
      gp.fit(df_sampled.loc[:,("lat", "lng")], df_sampled["true"] - df_sampled["pred"])
      preds = df_val["pred"] + gp.predict(df_val.loc[:,("lat", "lng")])
      logger.tick(df_val, preds)
    logger.clear_run()
  df_train = pd.concat([df_train, df_val], axis=0)
  for _ in range(n_reps):
    idxs_random = np.arange(len(df_train))
    np.random.shuffle(idxs_random)
    for i in tqdm(range(1, len(df_train) - 1)):
      df_sampled = df_train.iloc[idxs_random[:i],:]
      df_rest = df_train.iloc[np.setdiff1d(idxs_random, idxs_random[:i]),:]
      gp = GaussianProcessRegressor(kernel=isotropic_kernel, normalize_y=True)
      gp.fit(df_sampled.loc[:,("lat", "lng")], df_sampled["true"] - df_sampled["pred"])
      preds = df_rest["pred"] + gp.predict(df_rest.loc[:,("lat", "lng")])
      coeff_logger.tick(df_sampled, df_rest, preds)
    coeff_logger.clear_run()
  return logger, coeff_logger

def sample_random_no_sat(df_train, df_val, n_reps=10):
  """
  Randomly sample, without building on top of satellite predictions.
  """
  logger = SamplingLogger("random_nosat")
  coeff_logger = CoeffSamplingLogger("random_nosat")
  for _ in range(n_reps):
    idxs_random = np.arange(len(df_train))
    np.random.shuffle(idxs_random)
    for i in tqdm(range(1, len(df_train))):
      df_sampled = df_train.iloc[idxs_random[:i],:]
      gp = GaussianProcessRegressor(kernel=isotropic_kernel, normalize_y=True)
      gp.fit(df_sampled.loc[:,("lat", "lng")], df_sampled["true"])
      preds = gp.predict(df_val.loc[:,("lat", "lng")])
      logger.tick(df_val, preds)
    logger.clear_run()
  df_train = pd.concat([df_train, df_val], axis=0)
  for _ in range(n_reps):
    idxs_random = np.arange(len(df_train))
    np.random.shuffle(idxs_random)
    for i in tqdm(range(1, len(df_train) - 1)):
      df_sampled = df_train.iloc[idxs_random[:i],:]
      df_rest = df_train.iloc[np.setdiff1d(idxs_random, idxs_random[:i]),:]
      gp = GaussianProcessRegressor(kernel=isotropic_kernel, normalize_y=True)
      gp.fit(df_sampled.loc[:,("lat", "lng")], df_sampled["true"])
      preds = gp.predict(df_rest.loc[:,("lat", "lng")])
      coeff_logger.tick(df_sampled, df_rest, preds)
    coeff_logger.clear_run()
  return logger, coeff_logger


if __name__ == "__main__":

  arg_parser = ArgumentParser()
  arg_parser.add_argument("--fold_idx", type=str, default=None)
  args = arg_parser.parse_args()
  fold_idx = args.fold_idx

  electrification_data = pd.read_csv("../data/electrification.csv")
  df = pd.read_csv("../results/fold_%s/test_results.csv" % fold_idx)
  print("Length of original test set: %d" % len(df))

  df = df.merge(electrification_data, how="inner", left_on="id", right_on="village_id")
  df = df.loc[:,("smoothed", "true", "lat", "lng", "taluk_idx", "district_idx", "electrification")]
  df = df.rename(columns={"smoothed": "pred"})

  print("Length of merged test set: %d" % len(df))

  df = df.groupby("taluk_idx").mean().loc[:,("pred", "true", "lat", "lng", "electrification")]
  print("Shape of aggregated test set:", df.shape)

  idxs = np.arange(len(df))
  train_idxs = np.random.choice(idxs, int(0.5 * len(idxs)), replace=False)
  val_idxs = np.array(list(set(idxs) - set(train_idxs)))
  df_train = df.iloc[train_idxs,:]
  df_val = df.iloc[val_idxs,:]

  logs, coeff_logs = sample_random_with_sat(df_train, df_val)
  logs.save(fold_idx)
  coeff_logs.save(fold_idx)
  logs, coeff_logs = sample_random_no_sat(df_train, df_val)
  logs.save(fold_idx)
  coeff_logs.save(fold_idx)
  logs, coeff_logs = sample_greedy_with_sat(df_train, df_val)
  logs.save(fold_idx)
  coeff_logs.save(fold_idx)
  logs, coeff_logs = sample_greedy_no_sat(df_train, df_val)
  logs.save(fold_idx)
  coeff_logs.save(fold_idx)

  df_train.to_csv("../results/fold_%s/sampling_train.csv" % fold_idx)
  df_val.to_csv("../results/fold_%s/sampling_val.csv" % fold_idx)
