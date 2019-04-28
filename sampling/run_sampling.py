import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from collections import Counter
from tqdm import tqdm
from dfply import *
from gp.gp import StochasticMeanGP, ConstantMeanGP
from utils import KERNEL, NOISE_LVL, B, GP_MAX_SIZE, SamplingLogger


def sample_random(df_train, df_test, h_train, h_test, gp_factory,
                  max_samples=500, n_reps=5, log_freq=50):
    """
    Randomly sample villages.
    """
    print(f"== Sampling randomly for {max_samples} samples.")
    logger = SamplingLogger("random")
    for _ in range(n_reps):
        idxs_random = np.random.choice(df_train.shape[0], max_samples, False)
        for i in tqdm(range(1, max_samples + 1)):
            if i % log_freq == 0:
                df_sampled = df_train >> row_slice(idxs_random[:i])
                gp = gp_factory()
                gp.fit(df_sampled[["lat","lng"]].values,
                       df_sampled["true"].values,
                       h_train[idxs_random[:i], :])
                preds, _ = gp.predict(df_test[["lat", "lng"]].values, h_test)
                logger.tick(i, df_test, preds)
        logger.clear_run(idxs_random)
    return logger


def sample_random_no_sat(df_train, df_test, gp_factory,
                         max_samples=500, n_reps=5, log_freq=50):
    """
    Randomly sample villages but don't use satellite imagery when building GPs.
    """
    print(f"== Sampling randomly for {max_samples} samples.")
    logger = SamplingLogger("no_sat")
    for _ in range(n_reps):
        idxs_random = np.random.choice(df_train.shape[0], max_samples, False)
        for i in tqdm(range(1, max_samples + 1)):
            if i % log_freq == 0:
                df_sampled = df_train >> row_slice(idxs_random[:i])
                gp = gp_factory()
                gp.fit(df_sampled[["lat","lng"]].values,
                       df_sampled["true"].values)
                preds, _ = gp.predict(df_test[["lat", "lng"]].values)
                logger.tick(i, df_test, preds)
        logger.clear_run(idxs_random)
    return logger


def sample_with_prior_var(df_train, df_test, h_train, h_test, gp_factory,
                          max_samples=500, n_reps=5, log_freq=50):
    """
    Randomly sample villages proportional to prior variance.
    """
    print(f"== Sampling proportional to prior var for {max_samples} samples.")
    logger = SamplingLogger("prior_var")
    for _ in range(n_reps):
        gp = gp_factory()
        _, var = gp.predict(df_train[["lat", "lng"]].values, h_train)
        idxs = np.random.choice(df_train.shape[0], max_samples, False,
                                p=np.diag(var) / np.sum(np.diag(var)))
        for i in tqdm(range(1, max_samples + 1)):
            if i % log_freq == 0:
                df_sampled = df_train >> row_slice(idxs[:i])
                gp = gp_factory()
                gp.fit(df_sampled[["lat","lng"]].values,
                       df_sampled["true"].values,
                       h_train[idxs[:i], :])
                preds, _ = gp.predict(df_test[["lat", "lng"]].values, h_test)
                logger.tick(i, df_test, preds)
        logger.clear_run(idxs)
    return logger


def sample_sequential(df_train, df_test, h_train, h_test, gp_factory,
                      max_samples=500, n_reps=5, log_freq=50):
    """
    Randomly sample villages sequentially, picking out the village with maximum
    variance in the GP at each step.
    """
    print(f"== Sampling proportional to prior var for {max_samples} samples.")
    print(f"== Limiting training set size to {GP_MAX_SIZE}.")
    logger = SamplingLogger("sequential")
    for _ in range(n_reps):
        gp = gp_factory()
        _, var = gp.predict(df_train[["lat", "lng"]].values, h_train)
        idxs = np.array([np.argmax(np.diag(var))])
        all_idxs = np.random.choice(np.arange(len(train_idxs)), GP_MAX_SIZE,
                                    replace=False,
                                    p=np.diag(var) / np.sum(np.diag(var)))
        for i in tqdm(range(1, max_samples + 1)):
            gp.fit(df_train[["lat","lng"]].values[idxs, :],
                   df_train["true"].values[idxs],
                   h_train[idxs, :])
            remaining_idxs = np.array(list(set(all_idxs) - set(idxs)))
            mean, var = gp.predict(
                    df_train[["lat", "lng"]].values[remaining_idxs,:],
                    h_train[remaining_idxs,:])
            var[np.isnan(var)] = 0
            idxs = np.append(idxs, remaining_idxs[np.argmax(np.diag(var))])
            if i % log_freq == 0:
                df_sampled = df_train >> row_slice(idxs[:i])
                gp = gp_factory()
                gp.fit(df_sampled[["lat","lng"]].values,
                       df_sampled["true"].values,
                       h_train[idxs[:i], :])
                preds, _ = gp.predict(df_test[["lat", "lng"]].values, h_test)
                logger.tick(i, df_test, preds)
        logger.clear_run(idxs)
    return logger


def sample_unexplored(df_train, df_test, h_train, h_test, gp_factory,
                      max_samples=500, n_reps=5, log_freq=50):
    """
    Randomly sample villages sequentially, picking out a taluk with the
    highest fraction of unexplored villages at each step.
    """
    print(f"== Sampling proportional to prior var for {max_samples} samples.")
    logger = SamplingLogger("prior_var")
    for _ in range(n_reps):
        total_counts = Counter(df_train["taluk_idx"])
        all_idxs = np.arange(df_train.shape[0])
        idxs = np.array([np.random.choice(df_train.shape[0])])
        for i in tqdm(range(1, max_samples + 1)):
            remaining_idxs = np.array(list(set(all_idxs) - set(idxs)))
            remaining_counts = total_counts - Counter(df_trai["taluk_idx"][idxs])
            taluks = np.zeros(max(total_counts.keys()) + 1)
            for k, v in remaining_counts.items():
                taluks[k] = v / total_counts[k]
            taluk_chosen = np.argmax(taluks)
            mask = df_train.iloc[remaining_idxs]["taluk_idx"] == taluk_chosen
            chosen_idx = np.random.choice(remaining_idxs, p=mask * remaining_idxs / sum(mask * remaining_idxs))
            idxs = np.append(idxs, chosen_idx)
            if i % log_freq == 0:
                df_sampled = df_train >> row_slice(idxs[:i])
                gp = gp_factory()
                gp.fit(df_sampled[["lat","lng"]].values,
                       df_sampled["true"].values,
                       h_train[idxs[:i], :])
                preds, _ = gp.predict(df_test[["lat", "lng"]].values, h_test)
                logger.tick(i, df_test, preds)
        logger.clear_run(idxs)
    return logger


if __name__ == "__main__":

    np.random.seed(123)

    argparser = ArgumentParser()
    argparser.add_argument("--data-subdir", type=str, default=None)
    argparser.add_argument("--max-samples", type=int, default=500)
    argparser.add_argument("--n-reps", type=int, default=5)
    argparser.add_argument("--log-freq", type=int, default=50)
    argparser.add_argument("--split-data", action="store_true")
    argparser.add_argument("--strategy", type=str, default="random")
    args = argparser.parse_args()

    # load in the relevant data
    df = pd.read_csv(f"results/{args.data_subdir}/test_df.csv")
    mu = np.load(f"data/{args.data_subdir}/mu.npy").item()
    std = np.load(f"data/{args.data_subdir}/std.npy").item()
    df["pred"] = (df["pred"] - mu) / std
    df["true"] = (df["true"] - mu) / std
    final_layer = np.load(f"results/{args.data_subdir}/final_layer.npy")
    final_layer_weight = np.load(f"results/{args.data_subdir}/final_layer_weight.npy")
    final_layer_bias = np.load(f"results/{args.data_subdir}/final_layer_bias.npy")

    print("Running for state: ", df["state_name"][0])
    print(f"Length of test set: {len(df)}")

    # partition train/test split
    if args.split_data:
        K = min(5000, int(df.shape[0] * 1/3))
        train_idxs, test_idxs = train_test_split(np.arange(df.shape[0]),
                                                 test_size=K,
                                                 stratify=df["district_idx"])
        np.save(f"results/{args.data_subdir}/train_idxs.npy", train_idxs)
        np.save(f"results/{args.data_subdir}/test_idxs.npy", test_idxs)
    else:
        train_idxs = np.load(f"results/{args.data_subdir}/train_idxs.npy")
        test_idxs = np.load(f"results/{args.data_subdir}/test_idxs.npy")

    # set up dataset
    final_layer = np.c_[final_layer, np.ones(len(final_layer))]
    df_train = df >> row_slice(train_idxs)
    df_test = df >> row_slice(test_idxs)
    h_train = final_layer[train_idxs]
    h_test = final_layer[test_idxs]
    print("Train:", df_train.shape)
    print("Test:", df_test.shape)

    gp_factory = lambda: StochasticMeanGP(
            np.r_[final_layer_weight, final_layer_bias], B, KERNEL, NOISE_LVL)
    gp_no_sat_factory = lambda: ConstantMeanGP(0, KERNEL, NOISE_LVL)

    if args.strategy == "random":
        logs = sample_random(df_train, df_test, h_train, h_test, gp_factory,
                             args.max_samples, args.n_reps, args.log_freq)
    elif args.strategy == "prior_var":
        logs = sample_with_prior_var(df_train, df_test, h_train, h_test,
                                     gp_factory, args.max_samples, args.n_reps,
                                     args.log_freq)
    elif args.strategy == "sequential":
        logs = sample_sequential(df_train, df_test, h_train, h_test, gp_factory,
                                 args.max_samples, args.n_reps, args.log_freq)
    elif args.strategy == "no_sat":
        logs = sample_random_no_sat(df_train, df_test, gp_no_sat_factory, 
                                    args.max_samples, args.n_reps, 
                                    args.log_freq)
    else:
        raise ValueError("Strategy is incorrect.")

    logs.save(args.data_subdir)
