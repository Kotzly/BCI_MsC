from collections import OrderedDict
from .base import LightiningEEGModule
from .utils import PoolParams, cast_or_default
from torch import nn
from .normalization import apply_max_norm
import torch


class DeepConvNet(LightiningEEGModule):

    DEFAULT_MAX_POOL = PoolParams((1, 3), (1, 3))
    DEFAULT_CONV_LENGTH = 10

    def __init__(
        self,
        n_channels,
        n_classes,
        length,
        *,
        temporal_filters_length=10,
        n_temporal_filters=25,
        n_spatial_filters=25,
        n_filters=None,
        max_pool_1=None,
        max_pool_2=None,
        max_pool_3=None,
        max_pool_4=None,
        conv_length_2=None,
        conv_length_3=None,
        conv_length_4=None,
        p=.5
    ):
        # N C, H, W
        super(DeepConvNet, self).__init__()

        n_filters = n_filters or (50, 100, 200)
        if not isinstance(n_filters, (tuple, list)):
            raise Exception("'n_filters' must be a tuple or a list")
        f1, f2, f3 = n_filters

        max_pool_1 = cast_or_default(max_pool_1, PoolParams, self.DEFAULT_MAX_POOL)
        max_pool_2 = cast_or_default(max_pool_2, PoolParams, self.DEFAULT_MAX_POOL)
        max_pool_3 = cast_or_default(max_pool_3, PoolParams, self.DEFAULT_MAX_POOL)
        max_pool_4 = cast_or_default(max_pool_4, PoolParams, self.DEFAULT_MAX_POOL)
        conv_length_2 = conv_length_2 or self.DEFAULT_CONV_LENGTH
        conv_length_3 = conv_length_3 or self.DEFAULT_CONV_LENGTH
        conv_length_4 = conv_length_4 or self.DEFAULT_CONV_LENGTH

        n_classes = 1 if (n_classes <= 2) else n_classes

        self.conv_pool_block_1 = nn.Sequential(
            OrderedDict(
                temporal_filter=nn.Conv2d(
                    1, n_temporal_filters, (1, temporal_filters_length), groups=1, padding="valid"
                ),
                spatial_filter=nn.Conv2d(
                    n_temporal_filters, n_spatial_filters, (n_channels, 1), stride=1, groups=1, padding="valid", bias=False
                ),
                batch_norm=nn.BatchNorm2d(n_spatial_filters, 1e-5, momentum=.1),
                non_linearity=nn.ELU(),
                max_pool=nn.MaxPool2d(**max_pool_1._asdict(), ceil_mode=False)
            )
        )

        self.conv_pool_block_2 = nn.Sequential(
            OrderedDict(
                conv=nn.Conv2d(
                    n_spatial_filters, f1, (1, conv_length_2), stride=1, groups=1, padding="valid",
                ),
                non_linearity=nn.ELU(),
                batch_norm=nn.BatchNorm2d(f1, 1e-5, momentum=.1),
                max_pool=nn.MaxPool2d(**max_pool_2._asdict(), ceil_mode=False)
            )
        )
        self.conv_pool_block_3 = nn.Sequential(
            OrderedDict(
                conv=nn.Conv2d(
                    f1, f2, (1, conv_length_3), stride=1, groups=1, padding="valid",
                ),
                non_linearity=nn.ELU(),
                batch_norm=nn.BatchNorm2d(f2, 1e-5, momentum=.1),
                max_pool=nn.MaxPool2d(**max_pool_3._asdict(), ceil_mode=False)
            )
        )
        self.conv_pool_block_4 = nn.Sequential(
            OrderedDict(
                conv=nn.Conv2d(
                    f2, f3, (1, conv_length_4), stride=1, groups=1, padding="valid",
                ),
                non_linearity=nn.ELU(),
                batch_norm=nn.BatchNorm2d(f3, 1e-5, momentum=.1),
                max_pool=nn.MaxPool2d(**max_pool_4._asdict(), ceil_mode=False)
            )
        )

        extractor = nn.Sequential(
            self.conv_pool_block_1,
            self.conv_pool_block_2,
            self.conv_pool_block_3,
            self.conv_pool_block_4,
        )
        final_size = extractor(torch.empty(1, 1, n_channels, length)).size(3)
        del extractor

        self.classifier = nn.Sequential(
            OrderedDict(
                flatten=nn.Flatten(),
                linear=nn.Linear(f3 * final_size, n_classes)
            )
        )

        self.dropout_1 = nn.Dropout(p=p)
        self.dropout_2 = nn.Dropout(p=p)
        self.dropout_3 = nn.Dropout(p=p)
        self.dropout_4 = nn.Dropout(p=p)

    def apply_constraints(self):
        apply_max_norm(self.conv_pool_block_1.temporal_filter.weight, dim=3, max_value=2.)
        apply_max_norm(self.conv_pool_block_1.spatial_filter.weight, dim=2, max_value=2.)
        apply_max_norm(self.conv_pool_block_2.conv.weight, dim=3, max_value=2.)
        apply_max_norm(self.conv_pool_block_3.conv.weight, dim=3, max_value=2.)
        apply_max_norm(self.conv_pool_block_4.conv.weight, dim=3, max_value=2.)
        apply_max_norm(self.classifier.linear.weight, dim=1, max_value=.5)

    def forward(self, x):
        if self.training:
            self.apply_constraints()
        output = self.conv_pool_block_1(x)
        output = self.dropout_1(output)
        output = self.conv_pool_block_2(output)
        output = self.dropout_2(output)
        output = self.conv_pool_block_3(output)
        output = self.dropout_3(output)
        output = self.conv_pool_block_4(output)
        output = self.dropout_4(output)
        output = self.classifier(output)

        return output
