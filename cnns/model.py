import torch
import torch.nn as nn
import pretrainedmodels


def get_pretrained_firstconv_weights():
    return torch.load("cnns/firstconv_weights.torch")


class CombinedImageryCNN(nn.Module):

    def __init__(self, initialize=False):
        super().__init__()
        self.base = pretrainedmodels.__dict__["resnet18"](num_classes=1000,
                pretrained="imagenet" if initialize else None)
        self.base.conv1= nn.Conv2d(6, 64, kernel_size=(7, 7),
                                   stride=(2, 2), padding=(3, 3),
                                   bias=False)
        self.final_fc = nn.Linear(self.base.last_linear.out_features, 1)


    def initialize_weights(self):
        weights = get_pretrained_firstconv_weights()
        self.base.conv1.weight.data = torch.cat((weights, weights), dim=1)

    def forward(self, x):
        return self.final_fc(self.base(x))

    def get_final_layer(self, x):
        return self.base(x)

    def get_final_layer_weights(self):
        return self.final_fc.weight.data[0], self.final_fc.bias.data
