import numpy as np
import pandas as pd
import pickle
from argparse import ArgumentParser
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm
from utils import tuned_kernel, SamplingLogger


def sample_random(df, pred_strategy="with_sat", max_samples=1000, n_reps=3, 
                  batch_size=10, tune_freq=5):
    """
    Randomly sample villages and make predictions for the entire dataset.
    """
    if pred_strategy not in ("with_sat", "no_sat"):
        raise ValueException("Prediction strategy invalid.")
    print(f"== Sampling {pred_strategy} for {max_samples} samples.")
    args = {"batch_size": batch_size, "tune_freq": tune_freq,
            "max_samples": max_samples}
    logger = SamplingLogger(pred_strategy, args)
    if pred_strategy == "with_sat":
        df["tgt"] = df["true"] - df["pred"]
        df["delta"] = df["pred"]
    else:
        df["tgt"] = df["true"]
        df["delta"] = 0
    for _ in range(n_reps):
        idxs_random = np.arange(len(df))
        np.random.shuffle(idxs_random)
        curr_kernel = tuned_kernel
        n_samples = min(max_samples, len(df))
        for i in tqdm(range(1, n_samples // batch_size + 1)):
            df_sampled = df.iloc[idxs_random[:i * batch_size], :]
            optimizer = "fmin_l_bfgs_b" if (i - 1) % tune_freq == 0 else None
            gp = GaussianProcessRegressor(kernel=curr_kernel, 
                                          normalize_y=True,
                                          optimizer=optimizer)
            gp.fit(df_sampled.loc[:, ("lat", "lng")], df_sampled["tgt"])
            curr_kernel = gp.kernel_
            preds = gp.predict(df.loc[:, ("lat", "lng")]) 
            logger.tick(i, df, preds + df["delta"])
        logger.clear_run(idxs_random)
    return logger


if __name__ == "__main__":

    np.random.seed(123)

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--fold-idx", type=str, default=None)
    arg_parser.add_argument("--max-samples", type=int, default=1000)
    arg_parser.add_argument("--batch-size", type=int, default=10)
    arg_parser.add_argument("--tune-freq", type=int, default=5)
    arg_parser.add_argument("--n-reps", type=int, default=3)
    args = arg_parser.parse_args()

    electrification_data = pd.read_csv("../data/electrification.csv")
    df = pd.read_csv(f"../results/fold_india/test_results.csv")
    print("Running for state: ", df["state_name"][0])
    print(f"Length of original test set: {len(df)}")
    df = df.rename(columns={"smoothed": "pred"})

    logs = sample_random(df, "with_sat", args.max_samples, args.n_reps,
                         args.batch_size, args.tune_freq)
    logs.save(args.fold_idx)
    logs = sample_random(df, "no_sat", args.max_samples, args.n_reps,
                         args.batch_size, args.tune_freq)
    logs.save(args.fold_idx)
