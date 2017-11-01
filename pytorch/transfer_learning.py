"""
Transfer Learning tutorial
==========================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

In this tutorial, you will learn how to train your network using
transfer learning. You can read more about the transfer learning at `cs231n
notes <http://cs231n.github.io/transfer-learning/>`__

Quoting this notes,

    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.

These two major transfer learning scenarios looks as follows:

-  **Finetuning the convnet**: Instead of random initializaion, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.

"""
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import argparse
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from load_dataset import BangladeshDataset
import matplotlib.pyplot as plt
import time
import os

plt.ion()   # interactive mode

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# The problem we're going to solve today is to train a model to classify
# **ants** and **bees**. We have about 120 training images each for ants and bees.
# There are 75 validation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.
#
# This dataset is a very small subset of imagenet.
#
# .. Note ::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_
#    and extract it to the current directory.

# Data augmentation and normalization for training
# Just normalization for validation
"""
PyTorch transfer learning tutorial transforms
==================================
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
"""

use_gpu = torch.cuda.is_available()
print("Using GPU:", use_gpu)

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# TODO: put data here
"""
For jpegs
-------------------
train_data_dir = '~/data/bangladesh_vis_jpgs/train/'
val_data_dir = '~/data/bangladesh_vis_jpgs/train/'
"""
train_data_dir = '~/tiffs'
val_data_dir = '~/tiffs'

train_bangladesh_csv_path = '~/predicting-poverty/data/bangladesh_2015_train.csv'
val_bangladesh_csv_path = '~/predicting-poverty/data/bangladesh_2015_valid.csv'


train_dataset = BangladeshDataset(csv_file=train_bangladesh_csv_path,
                                           root_dir=train_data_dir,
                                           transform=data_transforms['train'],
                                           sat_type="l8")
val_dataset = BangladeshDataset(csv_file=val_bangladesh_csv_path,
                                           root_dir=val_data_dir,
                                           transform=data_transforms['val'],
                                           sat_type="l8")

image_datasets = {'train': train_dataset, 'val': val_dataset}

dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=8)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion, optimizer, args, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_r2 = 0.0
    best_y_pred = []
    best_y_true = []

    losses = {'train': [], 'val': []}
    r2s = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print(time.ctime())
        print('-' * 10)

        y_true = []
        y_pred = []
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            # running_corrects = 0
            # Iterate over data.
            for i, data in enumerate(dataloders[phase]):
                # get the inputs
                inputs, labels = data

                y_true += labels.numpy().tolist()
 
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.float().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels.float())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                preds = outputs.data
                loss = criterion(outputs, labels)

                y_pred += preds.squeeze().cpu().numpy().tolist()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                print("Batch", i, "Loss:", loss.data[0])

                # statistics
                running_loss += loss.data[0]
                #running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / (dataset_sizes[phase] / 4)
            epoch_ro, epoch_p = pearsonr(y_pred, y_true)
            epoch_ro = epoch_ro ** 2
            #epoch_acc = running_corrects / dataset_sizes[phase]

            losses[phase].append(epoch_loss)
            r2s[phase].append(epoch_ro)

            print('{} Loss: {:.4f} R2: {:.4f}'.format(
                phase, epoch_loss, epoch_ro))

            # deep copy the model
            if phase == 'val' and epoch_ro > best_r2:
                best_r2 = epoch_ro
                best_y_pred = y_pred
                best_y_true = y_true
                if use_gpu:
                    model.cpu()
                best_model_wts = model.state_dict()
                if use_gpu:
                    model.cuda()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best R2: {:4f}'.format(best_r2))
    y_pred_filename = "~/models/epochs_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + "finetune_" + str(args.fine_tune) + "_ypred" + ".npy"
    y_true_filename = "~/models/epochs_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + "finetune_" + str(args.fine_tune) + "_ytrue" + ".npy"
    np.save(y_pred_filename, best_y_pred)
    np.save(y_true_filename, best_y_true)

    # save losses and r2s
    for k, v in losses.items():
        np.save("./losses_{}.npy".format(k), np.array(v))
    for k, v in r2s.items():
        np.save("./r2s_{}.npy".format(k), np.array(v))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for transfer-learning")

    main_arg_parser.add_argument("--epochs", type=int, default=16,
                                  help="number of training epochs, default is 16")
    main_arg_parser.add_argument("--fine-tune", type=bool, default=False,
                                  help="fine tune full network if true, otherwise just FC layer")
    main_arg_parser.add_argument("--save-model-dir", type=str, default="~/models",
                                  help="save best trained model in this directory")


    args = main_arg_parser.parse_args()
    print("Begin training")
    print("Train for {} epochs.".format(args.epochs))
    print("Fine tune full network: " + str(args.fine_tune))
    print("Save best model in " + args.save_model_dir)
    save_model_filename = "epochs_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + "finetune_" + str(args.fine_tune) + ".model"
    print(save_model_filename)
    print("====================================")
    print

    """
    Only in ipynb can we visualize.

    # Get a batch of training data
    inputs, classes = next(iter(dataloders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])
    """    

    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrained model and reset final fully connected layer.
    #
    model_conv = torchvision.models.resnet18(pretrained=True)

    ######################################################################
    # ConvNet as fixed feature extractor
    # ----------------------------------
    #
    # Here, we need to freeze all the network except the final layer. We need
    # to set ``requires_grad == False`` to freeze the parameters so that the
    # gradients are not computed in ``backward()``.
    #
    # You can read more about this in the documentation
    # `here <http://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
    #
    if not args.fine_tune:
        for param in model_conv.parameters():
            param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features

    # 1 since we are only predicting household expenditure
    model_conv.fc = nn.Linear(num_ftrs, 1)

    if use_gpu:
        model_conv = model_conv.cuda()

    criterion = nn.MSELoss()

    params = model_conv.parameters() if args.fine_tune else model_conv.fc.parameters()
    optimizer_conv = Adam(params, 1e-3)
    """
    optimizer_conv = optim.SGD(params, lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    """

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 15-25 min on CPU. On GPU though, it takes less than a
    # minute.
    #

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # In this scenario we only fine tune the final linear layer.
    # On CPU this will take about half the time compared to previous scenario.
    # This is expected as gradients don't need to be computed for most of the
    # network. However, forward does need to be computed.
    #

    model_conv = train_model(model_conv, criterion, optimizer_conv, args, num_epochs=args.epochs)

    save_model_filename = "epochs_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + "finetune_" + str(args.fine_tune) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(model_conv.state_dict(), save_model_path)

    ######################################################################
    #

    #visualize_model(model_conv)

    #plt.ioff()
    #plt.show()



if __name__ == "__main__":
    main()
