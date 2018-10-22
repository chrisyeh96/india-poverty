import numpy as np
import pandas as pd
import dill
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, DotProduct, \
                                             WhiteKernel


MAX_N_SAMPLES = 200

class SubspaceRBF(RBF):
  """
  Modification of the RBF to only act on specified dimensions.
  i.e. K(x, y) = exp(-1/2 || x[dims] -  y[dims]||^2 / l^2)
  """
  def __init__(self, dims, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
    super(SubspaceRBF, self).__init__(length_scale, length_scale_bounds)
    self.dims = dims

  def __call__(self, X, Y=None, eval_gradient=False):
    if Y is None:
      return super(SubspaceRBF, self).__call__(
        X[:,self.dims], None, eval_gradient=eval_gradient)
    else:
      return super(SubspaceRBF, self).__call__(
        X[:,self.dims], Y[:,self.dims], eval_gradient=eval_gradient)


class SubspaceDot(DotProduct):
  """
  Modification of the DotProduct kernel to only act on specified dimensions.
  i.e. K(x, y) = sigma^2 x[dims] ^\top y[dims]
  """
  def __init__(self, dims, sigma_0=1.0, sigma_0_bounds=(1e-5, 1e5)):
    super(SubspaceDot, self).__init__(sigma_0, sigma_0_bounds)
    self.dims = dims

  def __call__(self, X, Y=None, eval_gradient=False):
    if Y is None:
      return super(SubspaceDot, self).__call__(
        X[:,self.dims], None, eval_gradient=eval_gradient)
    else:
      return super(SubspaceDot, self).__call__(
        X[:,self.dims], Y[:,self.dims], eval_gradient=eval_gradient)


class SamplingLogger(object):
  """
  Logger to store results from adaptive sampling.
  """
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
    dill.dump(self, open("../results/fold_%s/sampling_%s.pkl" % (fold_idx, self.name), "wb"))


class CoeffSamplingLogger(object):
  """
  Logger to store results from coefficient sampling for effect estimation.
  """
  def __init__(self, name):
    self.name = name
    self.coeffs_matrix = []
    self.coeffs = []

  def tick(self, df_sampled):
    coeff = pearsonr(df_sampled["electrification"], df_sampled["true"])[0]
    self.coeffs.append(coeff)

  def clear_run(self):
    if self.coeffs:
      self.coeffs_matrix.append(self.coeffs)
    self.coeffs = []

  def save(self, fold_idx):
    dill.dump(self, open("../results/fold_%s/coeff_sampling_%s.pkl" % (fold_idx, self.name), "wb"))


def get_taluk_df(fold_df, labels=["true", "pred", "electrification"]):
  """
  Aggregate
  """
  cols = []
  for label in labels:
    fold_df["tmp"] = fold_df[label] * fold_df["pop"]
    col = fold_df.groupby("taluk_idx")["tmp"].sum() / fold_df.groupby("taluk_idx")["pop"].sum()
    cols.append(col.rename(label))
  cols.append(fold_df.groupby("taluk_idx")["lat"].mean())
  cols.append(fold_df.groupby("taluk_idx")["lng"].mean())
  return pd.concat(cols, axis=1).reset_index().rename(columns={"taluk_idx": "idx"})


coeff_est_kernel = ConstantKernel() * SubspaceRBF(dims=np.array([0, 1])) + \
                   ConstantKernel() * SubspaceRBF(dims=np.array([2]))

isotropic_kernel = WhiteKernel() + \
                   RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))

anisotropic_kernel = WhiteKernel() + \
                     SubspaceRBF(dims=np.array([0, 1]), length_scale=1.0,
                                 length_scale_bounds=(1e-3, 1e3)) + \
                     SubspaceRBF(dims=np.array([2]), length_scale=1.0,
                                 length_scale_bounds=(1e-3, 1e3))
