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

#from __future__ import print_function, division

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
from sklearn import metrics
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns

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

use_gpu = torch.cuda.is_available()
print("Using GPU:", use_gpu)
   
"""
For jpegs
-------------------
train_data_dir = '~/data/bangladesh_vis_jpgs/train/'
val_data_dir = '~/data/bangladesh_vis_jpgs/train/'
"""


"""
For Tony
------------
train_data_dir = '../../bucket_dump'
val_data_dir = '../../bucket_dump'

train_bangladesh_csv_path = '~/predicting-poverty/data/bangladesh_2015_train.csv'
val_bangladesh_csv_path = '~/predicting-poverty/data/bangladesh_2015_valid.csv'
"""



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


def load_dataset(train_bangladesh_csv_path, val_bangladesh_csv_path, 
                 train_data_dir, val_data_dir, sat_type="l8", year=2015):
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


    train_dataset = BangladeshDataset(csv_file=train_bangladesh_csv_path,
                                               root_dir=train_data_dir,
                                               transform=data_transforms['train'],
                                               sat_type=sat_type, year=year)
    val_dataset = BangladeshDataset(csv_file=val_bangladesh_csv_path,
                                               root_dir=val_data_dir,
                                               transform=data_transforms['val'],
                                               sat_type=sat_type, year=year)

    image_datasets = {'train': train_dataset, 'val': val_dataset}

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes)
    return dataloders, dataset_sizes







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


def train_model(model, criterion, optimizer, args, dataloders, dataset_sizes, num_epochs=25,):
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

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            y_true = []
            y_pred = []
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

                #print("Batch", i, "Loss:", loss.data[0])

                # statistics
                running_loss += loss.data[0]
                #running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]

            #epoch_ro, epoch_p = pearsonr(y_pred, y_true)
            #epoch_ro = epoch_ro ** 2
            #epoch_acc = running_corrects / dataset_sizes[phase]

            epoch_r2 = metrics.r2_score(y_true, y_pred)

            losses[phase].append(epoch_loss)
            r2s[phase].append(epoch_r2)

            print('{} Loss: {:.4f} R2: {:.4f}'.format(
                phase, epoch_loss, epoch_r2))

            # deep copy the model
            if phase == 'val' and epoch_r2 > best_r2:
                best_r2 = epoch_r2
                best_y_pred = y_pred
                best_y_true = y_true
                if use_gpu:
                    model.cpu()
                best_model_wts = model.state_dict()
                if use_gpu:
                    model.cuda()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best R2: {:4f}'.format(best_r2))
    print(losses)
    print(r2s)
    
    
    """
    fig, ax = plt.subplots()
    #fig = plt.figure()
    #ax = plt.subplot(111)
    plt.plot(np.array(losses["train"]), label="train")
    plt.plot(np.array(losses["val"]), label="val")
    plt.title("LandSat-8: Loss over epochs")
    plt.xlabel("Epochs")
    plt.legend()
    #plt.show()

    plt.savefig('/home/echartock03/models/losses.png')

    
    fig, ax = plt.subplots()
    #fig = plt.figure()
    #ax = plt.subplot(111)
    plt.plot(np.array(r2s["train"]), label="train")
    plt.plot(np.array(r2s["val"]), label="val")
    plt.title("LandSat-8: $R^2$ over epochs")
    plt.xlabel("Epochs")
    plt.legend()
    #plt.show()

    plt.savefig('/home/echartock03/models/r2s.png')
    


    
    fig = sns.jointplot(best_y_true[np.logical_and(best_y_true > 0, best_y_pred > 0)], 
                    best_y_pred[np.logical_and(best_y_true > 0, best_y_pred > 0)], 
                    kind="reg", size=8, marker=".")
    fig.fig.set_size_inches((12, 8))
    fig.ax_joint.set(ylabel="Predicted household expenditures [Taka]",
                     xlabel="Observed household expenditures [Taka]")
    fig.ax_joint.set_title("Bangladesh: CNN predicted versus observed household expenditures", y=0.98);
    fig.savefig('/home/echartock03/models/predicted_vs_observed.png')
    """
    
    
    y_pred_filename = "/home/tony/models/{}_{}_bestypred_epochs_{}_finetune_{}.npy".format(args.sat_type, str(args.year), str(args.epochs), str(args.fine_tune))
    y_true_filename = "/home/tony/models/{}_{}_bestytrue_epochs_{}_finetune_{}.npy".format(args.sat_type, str(args.year), str(args.epochs), str(args.fine_tune))
    np.save(y_pred_filename, best_y_pred)
    np.save(y_true_filename, best_y_true)

    # save losses and r2s
    for k, v in losses.items():
        np.save("/home/tony/models/{}_{}_losses_{}_epochs_{}_finetune_{}.npy".format(args.sat_type, str(args.year), k, str(args.epochs), str(args.fine_tune)), np.array(v))
    for k, v in r2s.items():
        np.save("/home/tony/models/{}_{}_r2s_{}_epochs_{}_finetune_{}.npy".format(args.sat_type, str(args.year), k, str(args.epochs), str(args.fine_tune)), np.array(v))

    # load best model weights
    model.cpu()
    model.load_state_dict(best_model_wts)
    return model


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for transfer-learning")

    main_arg_parser.add_argument("--epochs", type=int, default=50,
                                  help="number of training epochs, default is 16")
    main_arg_parser.add_argument("--fine-tune", type=bool, default=False,
                                  help="fine tune full network if true, otherwise just FC layer")
    main_arg_parser.add_argument("--save-model-dir", type=str, default="/home/tony/models",
                                  help="save best trained model in this directory")
    main_arg_parser.add_argument("--sat-type", type=str, default="l8",
                                  help="l8 or s1")
    main_arg_parser.add_argument("--year", type=int, default=2015,
                                  help="2015 or 2011")


    args = main_arg_parser.parse_args()
    print("Begin training")
    print("Train for {} epochs.".format(args.epochs))
    print("Fine tune full network: " + str(args.fine_tune))
    print("Save best model in " + args.save_model_dir)
    print("Using satellite (type, year): " + args.sat_type + "," + str(args.year))
    save_model_filename = "epochs_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + "finetune_" + str(args.fine_tune) + ".model"
    print(save_model_filename)
    print("====================================")
    print

    """
    For Elliott
    ------------
    """
    train_data_dir = '/home/tony/bucket_files'
    val_data_dir = '/home/tony/bucket_files'

    train_bangladesh_csv_path = '../data/bangladesh_2015_train.csv'
    val_bangladesh_csv_path = '../data/bangladesh_2015_valid.csv'


    dataloders, dataset_sizes = load_dataset(train_bangladesh_csv_path, val_bangladesh_csv_path, 
                                             train_data_dir, val_data_dir, sat_type=args.sat_type, year=args.year)

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

    model_conv = train_model(model_conv, criterion, optimizer_conv, args, num_epochs=args.epochs, dataloders=dataloders,
                                                                          dataset_sizes=dataset_sizes)

    # save_model_filename defined above
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(model_conv.state_dict(), save_model_path)

    ######################################################################
    #

    #visualize_model(model_conv)

    #plt.ioff()
    #plt.show()



if __name__ == "__main__":
    main()
