from collections import OrderedDict
from .base import LightningEEGModule
from .utils import PoolParams, cast_or_default
from torch import nn
from .layers import Square, Log
from .normalization import apply_max_norm
import torch


class ShallowConvNet(LightningEEGModule):

    DEFAULT_AVG_POOL = PoolParams((1, 75), (1, 15))

    def __init__(
        self,
        n_channels,
        n_classes,
        length,
        *,
        temporal_filters_length=25,
        n_temporal_filters=40,
        n_spatial_filters=40,
        temporal_filter_max_norm=None,
        spatial_filter_max_norm=None,
        classifier_max_norm=None,
        avg_pool_params=None,
        p=.5,

    ):
        # N C, H, W
        super(ShallowConvNet, self).__init__()

        n_classes = 1 if (n_classes <= 2) else n_classes

        self.n_channels = n_channels
        self.length = length
        self.temporal_filter_max_norm = temporal_filter_max_norm or 2.
        self.spatial_filter_max_norm = spatial_filter_max_norm or 2.
        self.classifier_max_norm = classifier_max_norm or .5

        avg_pool_params = cast_or_default(avg_pool_params, PoolParams, self.DEFAULT_AVG_POOL)

        self.conv_block = nn.Sequential(
            OrderedDict(
                temporal_filter=nn.Conv2d(
                    1, n_temporal_filters, (1, temporal_filters_length), groups=1, padding="valid"
                ),
                spatial_filter=nn.Conv2d(
                    n_temporal_filters, n_spatial_filters, (n_channels, 1), stride=1, groups=1, padding="valid", bias=False
                ),
                batch_norm=nn.BatchNorm2d(n_spatial_filters, 1e-5, momentum=.1),
            )
        )

        self.square = Square()
        self.pool = nn.AvgPool2d(**avg_pool_params._asdict(), ceil_mode=False)
        self.logn = Log()

        final_size = self.get_final_size()

        self.classifier = nn.Sequential(
            OrderedDict(
                flatten=nn.Flatten(),
                dropout=nn.Dropout(p=p),
                linear=nn.Linear(n_spatial_filters * final_size, n_classes)
            )
        )

    def get_final_size(self):
        in_training = self.training
        self.eval()
        extractor = nn.Sequential(
            self.conv_block,
            self.pool
        )
        final_size = extractor(torch.empty(1, 1, self.n_channels, self.length)).size(3)
        if in_training:
            self.train()
        return final_size

    def apply_constraints(self):
        apply_max_norm(self.conv_block.temporal_filter.weight, dim=3, max_value=self.temporal_filter_max_norm)
        apply_max_norm(self.conv_block.spatial_filter.weight, dim=2, max_value=self.spatial_filter_max_norm)
        apply_max_norm(self.classifier.linear.weight, dim=1, max_value=self.classifier_max_norm)

    def forward(self, x):
        if self.training:
            self.apply_constraints()
        output = self.conv_block(x)
        output = self.square(output)
        output = self.pool(output)
        output = self.logn(output)
        output = self.classifier(output)
        return output
