from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn
import scipy as sp
import torch.optim as optim
import torchvision
import argparse
import time
import os
from sklearn import metrics
from scipy.stats import pearsonr
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, models, transforms
from tqdm import tqdm
from data import BangladeshDataset, IndiaDataset


home_dir = os.path.expanduser("~")
use_gpu = torch.cuda.is_available()
print("Using GPU:", use_gpu)


def load_dataset(test_csv_path, data_dir, country, label, sat_type="s1", year=2015, batch_size=128):
  if sat_type == "s1":
    sat_transforms = [transforms.CenterCrop(300), transforms.Resize(224)]
  else:
    sat_transforms = [transforms.CenterCrop(100), transforms.Resize(224)]
  data_transforms = {
    "test": transforms.Compose(sat_transforms + [
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ]),
  }
  if country == "india":
    dataset = IndiaDataset
  else:
    dataset = BangladeshDataset
  test_dataset = dataset(csv_file=test_csv_path,
                         root_dir=data_dir,
                         label=label,
                         transform=data_transforms["test"],
                         sat_type=sat_type, year=year)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                               num_workers=8, shuffle=False)
  return test_dataloader


def test_model(model, dataloader):

  y_true = []
  y_pred = []

  model.train(False)

  for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

    inputs, labels = data
    y_true += labels.numpy().tolist()

    if use_gpu:
      inputs = Variable(inputs.cuda())
      labels = Variable(labels.float().cuda())
    else:
      inputs, labels = Variable(inputs), Variable(labels.float())

    outputs = model(inputs)
    y_pred += outputs.data.squeeze().cpu().numpy().tolist()

  return y_true, y_pred


if __name__ == "__main__":

  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--country", type=str, default="india")
  arg_parser.add_argument("--label", type=str, default="secc_cons_per_cap_scaled")
  arg_parser.add_argument("--batch-size", type=int, default=128)
  arg_parser.add_argument("--data-subdir", type=str, default=None)
  arg_parser.add_argument("--model-name", type=str, default=None)
  arg_parser.add_argument("--verbose", action="store_true")
  arg_parser.add_argument("--save-results", action="store_true")
  arg_parser.set_defaults(fine_tune=True)

  args = arg_parser.parse_args()

  print("Begin testing for {}".format(args.country))
  print("Batch size {}".format(args.batch_size))
  print("====================================")
  print()

  data_dir = "{}/imagery".format(home_dir)

  if args.country == "india":

    test_csv_path = "../data/{}/test.csv".format(args.data_subdir)
    s1_data = load_dataset(test_csv_path, data_dir, "india", args.label, "s1", year=2015, batch_size=args.batch_size)
    l8_data = load_dataset(test_csv_path, data_dir, "india", args.label, "l8", year=2015, batch_size=args.batch_size)

  elif args.country == "bangladesh":

    test_csv_path = "../data/bangladesh_test.csv"
    s1_data = load_dataset(test_csv_path, data_dir, "bangladesh", "totexp_m_pc", "s1", year=2015, batch_size=args.batch_size)
    l8_data = load_dataset(test_csv_path, data_dir, "bangladesh", "totexp_m_pc", "l8", year=2015, batch_size=args.batch_size)

  model_s1 = torchvision.models.resnet18(pretrained=True)
  num_ftrs = model_s1.fc.in_features
  model_s1.fc = nn.Linear(num_ftrs, 1)
  model_path = "{}/predicting-poverty/models/{}_s1{}/saved_model.model".format(home_dir, args.model_name[:6], args.model_name[6:])
  model_s1.load_state_dict(torch.load(model_path))

  model_l8 = torchvision.models.resnet18(pretrained=True)
  model_l8.fc = nn.Linear(num_ftrs, 1)
  model_path = "{}/predicting-poverty/models/{}_l8{}/saved_model.model".format(home_dir, args.model_name[:6], args.model_name[6:])
  model_l8.load_state_dict(torch.load(model_path))

  if use_gpu:
    model_s1 = model_s1.cuda()
    model_l8 = model_l8.cuda()

  y_true_s1, y_pred_s1 = test_model(model_s1, s1_data)
  y_true_l8, y_pred_l8 = test_model(model_l8, l8_data)

  y_true_s1, y_true_l8 = np.array(y_true_s1), np.array(y_true_l8)
  y_pred_s1, y_pred_l8 = np.array(y_pred_s1), np.array(y_pred_l8)

  y_pred_joint = np.mean(np.array([y_pred_s1, y_pred_l8]), axis=0)

  print("S1:", metrics.r2_score(y_true_s1, y_pred_s1))
  print("L8:", metrics.r2_score(y_true_l8, y_pred_l8))
  print("Both:", metrics.r2_score(y_true_s1, y_pred_joint))
  print("No outliers:", metrics.r2_score(y_true_s1[y_true_s1 < 3], y_pred_joint[y_true_s1 < 3]))
  print("Correlation:", sp.stats.pearsonr(y_pred_l8, y_pred_s1))

  if args.save_results and args.country == "india":

    os.mkdir("../results/{}".format(args.model_name))
    np.save("../results/{}/y_true.npy".format(args.model_name), y_true_s1)
    np.save("../results/{}/y_pred_l8.npy".format(args.model_name), y_pred_l8)
    np.save("../results/{}/y_pred_s1.npy".format(args.model_name), y_pred_s1)

  if args.save_results and args.country == "bangladesh":

    os.mkdir("../results/bangladesh")
    np.save("../results/bangladesh/y_true.npy", y_true_s1)
    np.save("../results/bangladesh/y_pred_l8.npy", y_pred_l8)
    np.save("../results/bangladesh/y_pred_s1.npy", y_pred_s1)
