from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import argparse
import time
import os
from scipy.stats import pearsonr
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from sklearn import metrics
from load_dataset import BangladeshDataset, IndiaDataset


######################################################################
# Load Data
# ---------

home_dir = os.path.expanduser("~")
use_gpu = torch.cuda.is_available()
print("Using GPU:", use_gpu)


def load_dataset(train_csv_path, val_csv_path, train_data_dir, val_data_dir, country,
                 sat_type="l8", year=2015, batch_size=128):
    data_transforms = {
        "train": transforms.Compose([
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if country == "bangladesh":
        dataset = BangladeshDataset
    elif country == "india":
        dataset = IndiaDataset
    else:
        raise NotImplementedError("Nope")

    train_dataset = dataset(csv_file=train_csv_path,
                            root_dir=train_data_dir,
                            transform=data_transforms["train"],
                            sat_type=sat_type, year=year)
    val_dataset = dataset(csv_file=val_csv_path,
                          root_dir=val_data_dir,
                          transform=data_transforms["val"],
                          sat_type=sat_type, year=year)

    image_datasets = {"train": train_dataset, "val": val_dataset}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=8)
                  for x in ["train", "val"]}
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    print(dataset_sizes)
    return dataloaders, dataset_sizes


######################################################################
# Training the model
# ------------------

def train_model(model, criterion, optimizer, args, dataloaders, dataset_sizes, model_name, num_epochs=25):

    since = time.time()

    best_model_wts = model.state_dict()
    best_r2 = float("-inf")
    best_y_pred = []
    best_y_true = []

    losses = {"train": [], "val": []}
    r2s = {"train": [], "val": []}


    def save_logs(epoch_no=None):

        epoch_prefix = str(epoch_no) if epoch_no else ""
        if not os.path.exists(os.path.join(home_dir, "models/", model_name, epoch_prefix)):
            os.mkdir(os.path.join(home_dir, "models/", model_name, epoch_prefix))

        np.save(os.path.join(home_dir, "models/", model_name, epoch_prefix, "y_pred.npy"), best_y_pred)
        np.save(os.path.join(home_dir, "models/", model_name, epoch_prefix, "y_true.npy"), best_y_true)

        for k, v in losses.items():
            np.save(os.path.join(home_dir, "models/", model_name, epoch_prefix, "losses_{}.npy".format(k)),
                    np.array(v))
        for k, v in r2s.items():
            np.save(os.path.join(home_dir, "models/", model_name, epoch_prefix, "rsq_{}.npy".format(k)),
                    np.array(v))

        save_model_path = os.path.join(home_dir, "models/", model_name, epoch_prefix, "saved_model.model")
        torch.save(model.state_dict(), save_model_path)


    for epoch in range(1, num_epochs + 1):

        print("Epoch {}/{}".format(epoch, num_epochs))
        print(time.ctime())
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            y_true = []
            y_pred = []
            if phase == "train":
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

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

                if args.verbose:
                    #print("Batch", i, "Labels:", labels)
                    print("Batch", i, "Loss:", loss.data[0])

                running_loss += loss.data[0]

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_r2 = metrics.r2_score(y_true, y_pred)

            losses[phase].append(epoch_loss)
            r2s[phase].append(epoch_r2)

            print("{} Loss: {:.4f} R2: {:.4f}".format(phase, epoch_loss, epoch_r2))

            if phase == "val" and epoch_r2 > best_r2:

                best_r2 = epoch_r2
                best_y_pred = y_pred
                best_y_true = y_true
                if use_gpu:
                    model.cpu()
                best_model_wts = model.state_dict()
                if use_gpu:
                    model.cuda()

            if args.verbose and epoch % args.log_epoch_interval == 0:
                save_logs(epoch)

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
    time_elapsed // 60, time_elapsed % 60))
    print("Best R2: {:4f}".format(best_r2))

    save_logs()

    # load best model weights
    model.cpu()
    model.load_state_dict(best_model_wts)
    return model


def main():
    arg_parser = argparse.ArgumentParser(description="parser for transfer-learning")

    arg_parser.add_argument("--name", type=str, default=None,
                                  help="name for the model")
    arg_parser.add_argument("--epochs", type=int, default=50,
                                  help="number of training epochs, default is 16")
    arg_parser.add_argument("--fine-tune", type=bool, default=True,
                                  help="fine tune full network if true, otherwise just FC layer")
    arg_parser.add_argument("--country", type=str, default="bangladesh",
                                  help="bangladesh or india")
    arg_parser.add_argument("--sat-type", type=str, default="l8",
                                  help="l8 or s1")
    arg_parser.add_argument("--year", type=int, default=2015,
                                  help="2015 or 2011")
    arg_parser.add_argument("--batch_size", type=int, default=128)
    arg_parser.add_argument("--log_epoch_interval", type=int, default=20)
    arg_parser.add_argument("--verbose", action="store_true")

    args = arg_parser.parse_args()

    if not args.name:
        model_name = "{}_{}_{}_{}".format(
            args.country, args.sat_type, str(args.year), str(time.ctime()).replace(" ", "_"))
    else:
        model_name = args.name
    os.mkdir(os.path.join(home_dir, "models", model_name))

    print("Begin training for {}".format(args.country))
    print("Train for {} epochs".format(args.epochs))
    print("Batch size {}".format(args.batch_size))
    print("Fine tune full network: " + str(args.fine_tune))
    print("Save best model in: ~/models/{}".format(model_name))
    print("Using satellite (type, year): " + args.sat_type + "," + str(args.year))
    print("====================================")
    print()

    train_data_dir = "{}/data/{}_{}".format(home_dir, args.sat_type, args.year)
    val_data_dir = "{}/data/{}_{}".format(home_dir, args.sat_type, args.year)

    if args.country == "bangladesh":
        train_csv_path = "../data/bangladesh_2015_train.csv"
        val_csv_path = "../data/bangladesh_2015_valid.csv"
    elif args.country == "india":
        train_csv_path = "../data/india_train.csv"
        val_csv_path = "../data/india_valid.csv"
    else:
        raise NotImplementedError("Not implemented")

    dataloaders, dataset_sizes = load_dataset(train_csv_path, val_csv_path,
                                              train_data_dir, val_data_dir, args.country,
                                              sat_type=args.sat_type, year=args.year,
                                              batch_size=args.batch_size)

    model_conv = torchvision.models.resnet18(pretrained=True)

    if not args.fine_tune:
        for param in model_conv.parameters():
            param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 1)

    if use_gpu:
        model_conv = model_conv.cuda()

    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()

    params = model_conv.parameters() if args.fine_tune else model_conv.fc.parameters()
    optimizer_conv = optim.Adam(params, 1e-3)

    model_conv = train_model(model_conv, criterion, optimizer_conv, args, model_name=model_name, num_epochs=args.epochs, dataloaders=dataloaders,
                                                                          dataset_sizes=dataset_sizes)

    save_model_path = os.path.join(home_dir, "models/", model_name, "saved_model.model")
    torch.save(model_conv.state_dict(), save_model_path)


if __name__ == "__main__":
    main()
