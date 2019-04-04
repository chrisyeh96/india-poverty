import torch
import torch.nn as nn
import pretrainedmodels


class CombinedImageryCNN(nn.Module):

    def __init__(self, initialize=False):
        super().__init__()
        if initialize:
            base = pretrainedmodels.__dict__["se_resnext50_32x4d"](
                num_classes=1000, pretrained="imagenet")
            pretrained_conv_weight = model.layer0[0].weight
            base.layer0[0] = nn.Conv2d(6, 64, kernel_size=(7, 7),
                                       stride=(2, 2), padding=(3, 3),
                                       bias=False)
            base.layer0[0].weight[:,:3,:,:] = pretrained_conv_weight
            base.layer0[0].weight[:,3:,:,:] = pretrained_conv_weight
        else:
            base = pretrainedmodels.__dict__["se_resnext50_32x4d"](
                num_classes=1000, pretrained=None)
            base.layer0[0] = nn.Conv2d(6, 64, kernel_size=(7, 7),
                                       stride=(2, 2), padding=(3, 3),
                                       bias=False)
        self.base = base
        self.final_fc = nn.Linear(base.linear.out_features, 1, bias=True)

    def forward(self, x):
        return self.final_fc(self.base(x))
