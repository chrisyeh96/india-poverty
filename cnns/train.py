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
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, models, transforms
from data import IndiaDataset


home_dir = str(Path.home())
use_gpu = torch.cuda.is_available()
print("Using GPU:", use_gpu)


def load_dataset(train_csv_path, val_csv_path, train_data_dir, val_data_dir,
                 label, sat_type="s1", year=2015, batch_size=128,
                 train_frac=1.0):
    if sat_type == "s1":
        sat_transforms = [transforms.CenterCrop(300), transforms.Resize(224)]
    else:
        sat_transforms = [transforms.CenterCrop(100), transforms.Resize(224)]
    data_transforms = {
        "train": transforms.Compose(sat_transforms + [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose(sat_transforms + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    }

    dataset = IndiaDataset
    train_dataset = dataset(csv_file=train_csv_path,
                            root_dir=train_data_dir,
                            label=label,
                            transform=data_transforms["train"],
                            sat_type=sat_type, year=year, frac=train_frac)
    val_dataset = dataset(csv_file=val_csv_path,
                          root_dir=val_data_dir,
                          label=label,
                          transform=data_transforms["val"],
                          sat_type=sat_type, year=year, frac=train_frac)

    image_datasets = {
        "train": train_dataset,
        "val": val_dataset
    }

    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=batch_size,
                            num_workers=8, shuffle=True),
        "val": DataLoader(image_datasets["val"], batch_size=batch_size,
                          num_workers=8, shuffle=False)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    print("Dataset sizes", dataset_sizes)
    return dataloaders, dataset_sizes


def train_model(model, criterion, optimizer, dataloaders, dataset_sizes,
                model_name, num_epochs=25, verbose=False, log_epoch_interval=1):

    since = time.time()

    best_model_wts = model.state_dict()
    best_r2 = float("-inf")
    best_y_pred = []
    best_y_true = []

    losses = {"train": [], "val": []}
    r2s = {"train": [], "val": []}

    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=3, verbose=True)

    def save_logs(epoch_no=None):

        epoch_prefix = str(epoch_no) if epoch_no else ""
        base_dir = f"{home_dir}/predicting-poverty/models/{model_name}/{epoch_prefix}"
        Path(base_dir).mkdir(parents=True, exist_ok=True)

        np.save(f"{base_dir}/y_pred.npy"), best_y_pred)
        np.save(f"{base_dir}/y_true.npy"), best_y_true)

        for k, v in losses.items():
            np.save(f"{basedir}/losses_{k}.npy", np.array(v))
        for k, v in r2s.items():
            np.save(f"{base_dir}/rsq_{k}.npy", np.array(v))

        torch.save(model.state_dict(), f"{base_dir}/saved_model.model")

    for epoch in range(1, num_epochs + 1):

        print("Epoch {}/{}".format(epoch, num_epochs))
        print(time.ctime())
        print("=" * 10)

        for phase in ("train", "val"):
            y_true = []
            y_pred = []
            if phase == "train":
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            for i, data in enumerate(dataloaders[phase]):

                inputs, labels = data
                y_true += labels.numpy().tolist()

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.float().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels.float())

                optimizer.zero_grad()
                outputs = model(inputs)
                preds = outputs.data
                loss = criterion(outputs, labels)

                y_pred += preds.squeeze().cpu().numpy().tolist()

                if phase == "train":
                    loss.backward()
                    optimizer.step()
                if verbose:
                    print("Batch", i, "Loss:", loss.data[0])

                running_loss += loss.data[0]

            epoch_loss = running_loss / dataset_sizes[phase]
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
                    if use_gpu:
                        model.cpu()
                    best_model_wts = model.state_dict()
                    if use_gpu:
                        model.cuda()

            if verbose and epoch % log_epoch_interval == 0:
                save_logs(epoch)

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
    time_elapsed // 60, time_elapsed % 60))
    print("Best R2: {:4f}".format(best_r2))

    save_logs()
    model.cpu()
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--name", type=str, default=None)
    arg_parser.add_argument("--epochs", type=int, default=10)
    arg_parser.add_argument("--year", type=int, default=2015)
    arg_parser.add_argument("--label", type=str, default="log_secc_cons_per_cap_scaled")
    arg_parser.add_argument("--train-frac", type=float, default=1.0)
    arg_parser.add_argument("--lr", type=float, default=1e-5)
    arg_parser.add_argument("--weight-decay", type=float, default=0)
    arg_parser.add_argument("--batch-size", type=int, default=128)
    arg_parser.add_argument("--log-epoch-interval", type=int, default=1)
    arg_parser.add_argument("--preload-model", type=str, default=None)
    arg_parser.add_argument("--data-subdir", type=str, default=None)
    arg_parser.add_argument("--verbose", action="store_true")

    args = arg_parser.parse_args()

    if not args.name:
        model_name = "{}_{}".format(
            args.sat_type,
            str(args.year), str(time.ctime()).replace(" ", "_"))
    else:
        model_name = args.name
    os.mkdir(os.path.join(home_dir, "predicting-poverty/models", model_name))

    print(f"Train for {args.epochs} epochs")
    print(f"Batch size {args.batch_size}")
    print(f"Save best model in: ~/predicting-poverty/models/{model_name}")
    print("====================================\n")

    train_data_dir = f"{home_dir}/imagery"
    val_data_dir = f"{home_dir}/imagery"

    train_csv_path = f"../data/{args.data_subdir}/train.csv"
    val_csv_path = f"../data/{args.data_subdir}/valid.csv"

    dataloaders, dataset_sizes = load_dataset(train_csv_path, val_csv_path,
                                              train_data_dir, val_data_dir, args.label,
                                              sat_type=args.sat_type, year=args.year,
                                              batch_size=args.batch_size, train_frac=args.train_frac)

    model_conv = torchvision.models.resnet18(pretrained=True)

    if args.preload_model:
        model_path = f"{home_dir}/predicting-poverty/models/{args.preload_model}/saved_model.model"
        model_conv.load_state_dict(torch.load(model_path))

    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    if args.label == "log_secc_cons_per_cap_scaled":
        model_conv.fc = nn.Linear(num_ftrs, 1)
        criterion = nn.MSELoss()
    else:
        model_conv.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        criterion = nn.BCELoss()

    if use_gpu:
        model_conv = model_conv.cuda()

    params = model_conv.parameters()
    optimizer = optim.Adam(params, args.lr, weight_decay=args.weight_decay)

    model_conv = train_model(model_conv, criterion, optimizer,
                             model_name=model_name, num_epochs=args.epochs,
                             dataloaders=dataloaders, dataset_sizes=dataset_sizes,
                             verbose=args.verbose,
                             log_epoch_interval=args.log_epoch_interval)

    save_model_path = os.path.join(home_dir, "predicting-poverty/models/",
                                   model_name, "saved_model.model")
    torch.save(model_conv.state_dict(), save_model_path)
