from collections import OrderedDict
from .base import LightningEEGModule
from .normalization import apply_max_norm
from .layers import SeparableConv2d
from .init import glorot_weight_zero_bias
import torch
from torch import nn


class EEGNet(LightningEEGModule):
    def __init__(self, n_channels, n_classes, length, *, f1=4, d=2, f2=None, p=0.5):
        super(EEGNet, self).__init__()

        f2 = f2 or f1 * d

        self.f1 = f1
        self.d = d
        self.f2 = f2
        self.p = p
        self.n_channels = n_channels
        self.length = length

        n_classes = 1 if (n_classes <= 2) else n_classes

        depthwiseconv2d = nn.Conv2d(
            f1, d * f1, (n_channels, 1), groups=f1, padding="valid", bias=False
        )
        self.first_block_top = nn.Sequential(
            OrderedDict(
                conv2d=nn.Conv2d(1, f1, (1, 64), padding="same", bias=False),
                batchnorm=nn.BatchNorm2d(f1, 1e-5),
                depthwiseconv2d=depthwiseconv2d,
            )
        )

        self.first_block_bottom = nn.Sequential(
            OrderedDict(
                batchnorm=nn.BatchNorm2d(f1 * d, 1e-5),
                activation=nn.ELU(),
                avgpool2d=nn.AvgPool2d((1, 4), 4),
                dropout=nn.Dropout(p),
            )
        )

        self.second_block = nn.Sequential(
            OrderedDict(
                separableconv2d=SeparableConv2d(
                    d * f1,
                    f2,
                    kernel_size=(1, 16),
                    bias=False,
                    padding="same",
                    depth_multiplier=1,
                ),
                batchnorm=nn.BatchNorm2d(f2, 1e-5),
                activation=nn.ELU(),
                avgpool2d=nn.AvgPool2d((1, 8), 8),
                dropout=nn.Dropout(p),
            )
        )

        final_size = self.get_final_size()

        # Usage of torch.nn.CrossEntropyLoss does not need final softmax layer
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        self.classifier = nn.Sequential(
            OrderedDict(
                flatten=nn.Flatten(),
                # linear=nn.Linear(f2 * length // 32, n_classes),
                linear=nn.Linear(
                    final_size, n_classes
                ),  # If length is a power of 2, final size will be f2 * length // 32
            )
        )
        glorot_weight_zero_bias(self)

    def get_final_size(self):
        in_training = self.training
        self.eval()
        extractor = nn.Sequential(
            self.first_block_top,
            self.first_block_bottom,
            self.second_block,
            nn.Flatten(),
        )
        final_size = extractor(torch.empty(1, 1, self.n_channels, self.length)).size(1)
        if in_training:
            self.train()
        return final_size

    def apply_constraints(self, max_filter_norm=1.0, max_clf_norm=0.25):
        with torch.no_grad():
            apply_max_norm(
                self.first_block_top.depthwiseconv2d.weight,
                dim=2,
                max_value=max_filter_norm,
            )
            apply_max_norm(self.classifier.linear.weight, dim=1, max_value=max_clf_norm)

    def forward(self, x):
        if self.training:
            self.apply_constraints()
        output = self.first_block_top(x)
        output = self.first_block_bottom(output)
        output = self.second_block(output)
        output = self.classifier(output)
        return output
