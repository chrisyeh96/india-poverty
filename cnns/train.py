from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import argparse
import time
from pathlib import Path
from sklearn import metrics
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, models, transforms

from data import IndiaDataset, get_dataloader
from model import CombinedImageryCNN


home_dir = str(Path.home())
use_gpu = torch.cuda.is_available()
print("Using GPU:", use_gpu)


def train_model(model, criterion, optimizer, train_loader, val_loader,
                model_name, num_epochs=25, verbose=False, log_epoch_interval=1):

    since = time.time()

    best_model_wts = model.state_dict()
    best_r2 = float("-inf")
    best_y_pred = []
    best_y_true = []

    losses = {"train": [], "val": []}
    r2s = {"train": [], "val": []}

    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }

    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=3, verbose=True)

    def save_logs(epoch_no=None):

        epoch_prefix = str(epoch_no) if epoch_no else ""
        base_dir = f"models/{model_name}/{epoch_prefix}"
        Path(base_dir).mkdir(parents=True, exist_ok=True)

        np.save(f"{base_dir}/y_pred.npy", best_y_pred)
        np.save(f"{base_dir}/y_true.npy", best_y_true)

        for k, v in losses.items():
            np.save(f"{base_dir}/losses_{k}.npy", np.array(v))
        for k, v in r2s.items():
            np.save(f"{base_dir}/rsq_{k}.npy", np.array(v))

        torch.save(model.state_dict(), f"{base_dir}/saved_model.model")

    for epoch in range(1, num_epochs + 1):

        print(f"Epoch {epoch}/{num_epochs}")
        print(time.ctime())
        print("=" * 79)

        for phase in ("train", "val"):
            y_true = []
            y_pred = []
            if phase == "train":
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                y_true += labels.numpy().tolist()

                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                preds = outputs.data
                loss = criterion(outputs, labels)

                y_pred += preds.cpu().numpy().tolist()

                if phase == "train":
                    loss.backward()
                    optimizer.step()
                if verbose:
                    print(f"Batch {i}\tLoss: {loss.data.item():.2f}")

                running_loss += loss.data.item()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_r2 = metrics.r2_score(y_true, y_pred)

            losses[phase].append(epoch_loss)
            r2s[phase].append(epoch_r2)

            print(f"{phase} Loss: {epoch_loss:.4f} R2: {epoch_r2:.4f}")

            if phase == "val":
                scheduler.step(losses["val"][-1])
                if epoch_r2 > best_r2:
                    best_r2 = epoch_r2
                    best_y_pred = y_pred
                    best_y_true = y_true
                    best_model_wts = model.state_dict()

            if verbose and epoch % log_epoch_interval == 0:
                save_logs(epoch)

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
    time_elapsed // 60, time_elapsed % 60))
    print("Best R2: {:4f}".format(best_r2))

    save_logs()
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--name", type=str, default=None)
    arg_parser.add_argument("--epochs", type=int, default=10)
    arg_parser.add_argument("--label", type=str, default="log_secc_cons_per_cap_scaled")
    arg_parser.add_argument("--train-frac", type=float, default=1.0)
    arg_parser.add_argument("--lr", type=float, default=1e-5)
    arg_parser.add_argument("--weight-decay", type=float, default=0)
    arg_parser.add_argument("--batch-size", type=int, default=64)
    arg_parser.add_argument("--log-epoch-interval", type=int, default=1)
    arg_parser.add_argument("--preload-model", type=str, default=None)
    arg_parser.add_argument("--data-subdir", type=str, default=None)
    arg_parser.add_argument("--verbose", action="store_true")

    args = arg_parser.parse_args()

    if not args.name:
        model_name = f"{args.data_subdir}"
    else:
        model_name = args.name
    Path(f"models/{model_name}").mkdir(exist_ok=True, parents=True)

    print(f"Train for {args.epochs} epochs")
    print(f"Batch size {args.batch_size}")
    print(f"Save best model in: ~/predicting-poverty/models/{model_name}")
    print("=" * 79 + "\n")

    train_data_dir = f"{home_dir}/imagery"
    val_data_dir = f"{home_dir}/imagery"
    train_csv_path = f"data/{args.data_subdir}/train.csv"
    val_csv_path = f"data/{args.data_subdir}/valid.csv"

    train_loader = get_dataloader(train_csv_path, train_data_dir, args.label,
                                  batch_size=128, train=True, frac=1.0)
    val_loader = get_dataloader(val_csv_path, val_data_dir, args.label,
                                batch_size=128, train=True, frac=1.0)

    if args.preload_model:
        model_path = f"models/{args.preload_model}/saved_model.model"
        model = CombinedImageryCNN(initialize=False)
        model.load_state_dict(torch.load(model_path))
    else:
        model = CombinedImageryCNN(initialize=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    if not args.preload_model:
        model.initialize_weights()

    if use_gpu:
        model = model.cuda()

    model = train_model(model, criterion, optimizer,
                        model_name=model_name, num_epochs=args.epochs,
                        train_loader=train_loader, val_loader=val_loader,
                        verbose=args.verbose,
                        log_epoch_interval=args.log_epoch_interval)

    save_model_path = f"models/{model_name}/saved_model.model"
    torch.save(model.state_dict(), save_model_path)
