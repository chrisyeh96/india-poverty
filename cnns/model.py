import torch
import torch.nn as nn
import pretrainedmodels
from se_resnext import SENet, SEResNetBottleneck
from se_resnext import pretrained_settings, initialize_pretrained_model


class CombinedImageryCNN(nn.Module):

    def __init__(self):
        model = pretrainedmodels.__dict__["se_resnext50_32x4d"](num_classes=1000, pretrained='imagenet')
        pretrained_conv_weight = model.layer0[0].weight
        model.layer0[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.layer0[0].weight[:,:3,:,:] = pretrained_conv_weight
        model.layer0[0].weight[:,3:,:,:] = pretrained_conv_weight
        self.model = model

    def forward(self, x):
        return self.model(x)
