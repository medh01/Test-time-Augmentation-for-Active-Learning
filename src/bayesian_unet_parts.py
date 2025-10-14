import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """A block of two 2D convolutions, each followed by batch normalization and a ReLU activation.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class EncoderBloc(nn.Module):
    """An encoder block for the U-Net, consisting of a DoubleConv block, max pooling, and dropout.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        dropout_prob (float): The dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        x_conv = self.conv(x)
        x_down = self.pool(x_conv)
        x_down = self.dropout(x_down)
        return x_down, x_conv

class DecoderBloc(nn.Module):
    """A decoder block for the U-Net, which upsamples and concatenates with a skip connection.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
