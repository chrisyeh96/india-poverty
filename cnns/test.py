from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn
import scipy as sp
import argparse
import os
from sklearn import metrics
from scipy.stats import pearsonr
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import r2_score
from data import IndiaDataset, get_dataloader
from model import CombinedImageryCNN


home_dir = os.path.expanduser("~")
use_gpu = torch.cuda.is_available()
print("Using GPU:", use_gpu)


def test_model(model, dataloader):

    y_true = []
    y_pred = []
    final_layer = []

    model.train(False)

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

        inputs, labels = data

        if use_gpu:
            inputs = inputs.cuda()

        y_true += labels.numpy().tolist()
        y_pred += model(inputs).squeeze().data.cpu().numpy().tolist()
        final_layer += model.get_final_layer(inputs).data.cpu().numpy().tolist()

    return np.array(y_true), np.array(y_pred), np.array(final_layer)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--label", type=str, default="log_secc_cons_per_cap_scaled")
    arg_parser.add_argument("--batch-size", type=int, default=64)
    arg_parser.add_argument("--data-subdir", type=str, default=None)
    arg_parser.add_argument("--verbose", action="store_true")
    arg_parser.add_argument("--save-results", action="store_true")
    arg_parser.set_defaults(fine_tune=True)

    args = arg_parser.parse_args()

    print(f"Batch size {args.batch_size}")
    print("=" * 79 + "\n")

    test_csv_path = f"data/{args.data_subdir}/test.csv"
    data_dir = f"{home_dir}/imagery"

    test_loader = get_dataloader(test_csv_path, data_dir, args.label,
                                 batch_size=args.batch_size, train=False)

    model_path = f"models/{args.data_subdir}/saved_model.model"
    model = CombinedImageryCNN(initialize=False)
    model.load_state_dict(torch.load(model_path))

    if use_gpu:
        model = model.cuda()

    y_true, y_pred, final_layer = test_model(model, test_loader)
    print(f"R2: {r2_score(y_true, y_pred)}")
    Path(f"results/{args.data_subdir}").mkdir(exist_ok=True, parents=True)
    np.save(f"results/{args.data_subdir}/y_true.npy", y_true)
    np.save(f"results/{args.data_subdir}/y_pred.npy", y_pred)
    np.save(f"results/{args.data_subdir}/final_layer.npy", final_layer)
