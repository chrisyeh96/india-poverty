import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel


class SubspaceRBF(RBF):
    """
    Modification of the RBF to only act on specified dimensions.
    i.e. K(x, y) = exp(-1/2 || x[dims] -    y[dims]||^2 / l^2)
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
    Logger to store results from predicting poverty with sampling.
    """
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments
        self.r2s = []
        self.mses = []
        self.batch_nos = []
        self.idxs_matrix = []

    def tick(self, i, df, preds):
        self.r2s.append(r2_score(df["true"], preds))
        self.mses.append(mean_squared_error(df["true"], preds))
        self.batch_nos.append(i)

    def clear_run(self, idxs):
        self.idxs_matrix.append(idxs)

    def save(self, fold_idx):
        self.r2s = np.array(self.r2s)
        self.mses = np.array(self.mses)
        self.batch_nos = np.array(self.batch_nos)
        self.idxs_matrix = np.array(self.idxs_matrix)
        filename = f"../results/fold_{fold_idx}/sampling_{self.name}.pkl"
        pickle.dump(self, open(filename, "wb"))


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


isotropic_kernel = WhiteKernel() + \
                   RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))

anisotropic_kernel = WhiteKernel() + \
                     SubspaceRBF(dims=np.array([0, 1]), length_scale=1.0,
                                 length_scale_bounds=(1e-3, 1e3)) + \
                     SubspaceDot(dims=np.array([2]), sigma_0=1.0,
                                 sigma_0_bounds=(1e-3, 1e3))

tuned_kernel = WhiteKernel(noise_level=0.048) + RBF(length_scale=4.5)
