from __future__ import print_function, division

import numpy as np
import pandas as pd
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
from load_dataset import clean_household_data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import load_bangladesh_2015_tiff, load_india_tiff


######################################################################
# Load Data
# ---------

home_dir = os.path.expanduser("~")
use_gpu = torch.cuda.is_available()
print("Using GPU:", use_gpu)

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]


# for 2015 multiband data
class BangladeshMultibandDataset(Dataset):
    """Bangladesh Poverty dataset."""

    def __init__(self, csv_file, root_dir,sat_type="l8",mean=[0,0,0],std=[1,1,1],use_grouped_labels=False,target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.target_transform = target_transform
        self.sat_type = sat_type
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.households = clean_household_data(csv_file, sat_type)
        self.use_grouped_labels = use_grouped_labels
        self.grouped_labels = None
        if use_grouped_labels:
            self.grouped_labels = self.households.groupby("Village")["totexp_m_pc"].mean()
        
    def __len__(self):
        return len(self.households)


    def __getitem__(self, idx):
        hhid = self.households["a01"][idx]
        prefix = self.sat_type
        imgtype = "multiband"

        image = load_bangladesh_2015_tiff(self.root_dir, hhid, prefix, imgtype, quiet=True)
        # transpose makes shape image.shape = (500, 500, 3) #for multiband 6
        image = image.transpose((1, 2, 0))
        image = crop_center(image,224,224)
        image = (image.astype(np.float32) - self.mean)/self.std
        image = image.transpose((2,0,1))

        village = self.households["Village"][idx]
        if self.use_grouped_labels:
            village = self.households["Village"][idx]
            expenditure = self.grouped_labels[village]
        else:
            expenditure = self.households["totexp_m_pc"][idx]


        if self.target_transform:
            expenditure = self.target_transform(expenditure)

        return image, expenditure,village

    def get_grouped_labels(self):
        return self.grouped_labels




def load_dataset(train_csv_path, val_csv_path, train_data_dir, val_data_dir, country,
                 sat_type="l8", year=2015, batch_size=128, use_grouped_labels=False):


    dataset = BangladeshMultibandDataset
    multi_mean = [0.485, 0.456, 0.406,0,0,0]
    multi_std = [0.229, 0.224, 0.225,1,1,1]


    train_dataset = dataset(csv_file=train_csv_path,
                            root_dir=train_data_dir,
                            sat_type=sat_type,
                            mean = multi_mean,
                            std = multi_std, 
                            use_grouped_labels=use_grouped_labels)
    val_dataset = dataset(csv_file=val_csv_path,
                          root_dir=val_data_dir,
                          sat_type=sat_type,
                          mean = multi_mean,
                          std = multi_std, 
                          use_grouped_labels=use_grouped_labels)

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
        
    def exp_to_grouped_exp(y, villages):
        grouped_labels = pd.DataFrame({'Village': villages,
                                       'z_score': y })
        grouped_labels = grouped_labels.groupby(["Village"]).mean().reset_index()
        grouped_labels.sort_values(by='Village')
        return grouped_labels['z_score'].values        
        
    def calculate_grouped_r2(y_true, y_pred, epoch_villages):
        grouped_y_true = exp_to_grouped_exp(y_true, epoch_villages)
        grouped_y_pred = exp_to_grouped_exp(y_pred, epoch_villages)
        
        return metrics.r2_score(grouped_y_true, grouped_y_pred)
        


    for epoch in range(1, num_epochs + 1):

        print("Epoch {}/{}".format(epoch, num_epochs))
        print(time.ctime())
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            y_true = []
            y_pred = []
            epoch_villages = []
            if phase == "train":
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            for i, data in enumerate(dataloaders[phase]):

                inputs, labels, villages = data
                y_true += labels.numpy().tolist()
                epoch_villages += villages.numpy().tolist()
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.float().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels.float())

                optimizer.zero_grad()

                outputs = model(inputs.float())
                preds = outputs.data
                loss = criterion(outputs, labels)

                y_pred += preds.squeeze().cpu().numpy().tolist()

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                if args.verbose:
                    print("Batch", i, "Loss:", loss.data[0])

                running_loss += loss.data[0]

            epoch_loss = running_loss / dataset_sizes[phase]
            if args.use_grouped_labels:
                epoch_r2 = calculate_grouped_r2(y_true, y_pred, epoch_villages)
            else:
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
    arg_parser.add_argument("--epochs", type=int, default=10,
                                  help="number of training epochs, default is 10")
    arg_parser.add_argument("--fine-tune", type=bool, default=True,
                                  help="fine tune full network if true, otherwise only fc layer")
    arg_parser.add_argument("--country", type=str, default="bangladesh",
                                  help="bangladesh or india")
    arg_parser.add_argument("--sat-type", type=str, default="l8",
                                  help="l8 or s1")
    arg_parser.add_argument("--year", type=int, default=2015,
                                  help="2015 or 2011")
    arg_parser.add_argument("--lr", type=float, default=1e-4,
                                  help="learning rate")
    arg_parser.add_argument("--weight-decay", type=int, default=0,
                                  help="weight decay")
    arg_parser.add_argument("--batch-size", type=int, default=128,
                                  help="batch size")
    arg_parser.add_argument("--log-epoch-interval", type=int, default=50,
                                  help="how often to update epochs")
    arg_parser.add_argument("--preload-model", type=str, default=None,
                                  help="directory of stored model")
    arg_parser.add_argument("--use-grouped-labels", action="store_true")
    arg_parser.add_argument("--verbose", action="store_true")

    args = arg_parser.parse_args()

    if not args.name:
        model_name = "{}_{}_{}_{}".format(
            args.country, args.sat_type, str(args.year), str(time.ctime()).replace(" ", "_"))
    else:
        model_name = args.name
        
    if not os.path.isdir(os.path.join(home_dir, "models", model_name)):
        os.mkdir(os.path.join(home_dir, "models", model_name))

    print("Begin training for {}".format(args.country))
    print("Train for {} epochs".format(args.epochs))
    print("Batch size {}".format(args.batch_size))
    print("Fine tune full network: " + str(args.fine_tune))
    print("Save best model in: ~/models/{}".format(model_name))
    print("Using satellite (type, year): " + args.sat_type + "," + str(args.year))
    print("====================================")
    print()


    train_data_dir = '/home/hmishfaq/multiband2015_bd_tiff'
    val_data_dir = '/home/hmishfaq/multiband2015_bd_tiff'


    if args.country == "bangladesh":
        train_csv_path = '~/predicting-poverty/data/bangladesh_2015_train.csv' 
        val_csv_path = '~/predicting-poverty/data/bangladesh_2015_valid.csv' 
    elif args.country == "india":
        train_csv_path = "../data/india_train.csv"
        val_csv_path = "../data/india_valid.csv"
    else:
        raise NotImplementedError("Not implemented")

    dataloaders, dataset_sizes = load_dataset(train_csv_path, val_csv_path,
                                              train_data_dir, val_data_dir, args.country,
                                              sat_type=args.sat_type, year=args.year,
                                              batch_size=args.batch_size,
                                              use_grouped_labels=args.use_grouped_labels)

    model_conv = torchvision.models.resnet18(pretrained=True)
    if args.preload_model:
        model_conv.load_state_dict(torch.load("/home/echartock03/models/{}/saved_model.model".format(args.preload_model)))

    if not args.fine_tune:
        for param in model_conv.parameters():
            param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model_conv.fc = nn.Linear(num_ftrs, 1)

    if use_gpu:
        model_conv = model_conv.cuda()

    criterion = nn.SmoothL1Loss()

    params = model_conv.parameters() if args.fine_tune else model_conv.fc.parameters()
    optimizer_conv = optim.Adam(params, args.lr, weight_decay=args.weight_decay)

    model_conv = train_model(model_conv, criterion, optimizer_conv, args, model_name=model_name, num_epochs=args.epochs, dataloaders=dataloaders,
                                                                          dataset_sizes=dataset_sizes)

    save_model_path = os.path.join(home_dir, "models/", model_name, "saved_model.model")
    torch.save(model_conv.state_dict(), save_model_path)


if __name__ == "__main__":
    main()