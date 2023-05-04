from torch import nn
import torch


class SeparableConv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=False,
        padding="same",
        padding_mode='zeros',
        depth_multiplier=1,
    ):

        super(SeparableConv2d, self).__init__()
        # 16 * D * F1
        self.depthwise_layer = nn.Conv2d(
            in_channels,
            depth_multiplier * in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            dilation=1,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias
        )
        # F2 * D * F1
        self.pointwise_layer = nn.Conv2d(
            depth_multiplier * in_channels,
            out_channels,
            kernel_size=(1, 1),
            groups=1,
            dilation=1,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias
        )

    def forward(self, x):
        depthwise_output = self.depthwise_layer(x)
        pointwise_output = self.pointwise_layer(depthwise_output)
        return pointwise_output


class Reshape(nn.Module):

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


class ExpandDims(nn.Module):

    def __init__(self, dim=0):
        super(ExpandDims, self).__init__()
        self.dim = dim

    def forward(self, x):
        if x.ndim == 2:
            # For C, T
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            # For N, C, T
            x = x.unsqueeze(1)
        return x


class Square(nn.Module):

    def __init__(self):
        super(Square, self).__init__()

    def forward(self, x):
        return torch.square(x)


class Log(nn.Module):

    def __init__(self, eps=1e-6):
        super(Log, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, self.eps, None))
