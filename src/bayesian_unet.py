import torch.nn as nn
from bayesian_unet_parts import DoubleConv, EncoderBloc, DecoderBloc


class BayesianUNet(nn.Module):
    """A Bayesian U-Net model for semantic segmentation.

    This model implements the U-Net architecture with dropout layers in the encoder blocks,
    which allows for Monte Carlo dropout to be used for uncertainty estimation. The architecture
    consists of a contracting path (encoder), a bottleneck, and an expansive path (decoder).

    Args:
        in_channels (int): The number of input channels (e.g., 3 for RGB images).
        num_classes (int): The number of output classes for segmentation.
        dropout_prob (float): The dropout probability to be used in the encoder blocks.
    """
    def __init__(self, in_channels, num_classes, dropout_prob):
        super().__init__()

        # Encoder (Down sampling)
        self.encoder_bloc_1 = EncoderBloc(in_channels, 64, dropout_prob)
        self.encoder_bloc_2 = EncoderBloc(64, 128, dropout_prob)
        self.encoder_bloc_3 = EncoderBloc(128, 256, dropout_prob)
        self.encoder_bloc_4 = EncoderBloc(256, 512, dropout_prob)

        # Bottleneck
        self.bottle_neck = DoubleConv(512, 1024)

        # Decoder (Up sampling)
        self.decoder_bloc_1 = DecoderBloc(1024, 512)
        self.decoder_bloc_2 = DecoderBloc(512, 256)
        self.decoder_bloc_3 = DecoderBloc(256, 128)
        self.decoder_bloc_4 = DecoderBloc(128, 64)

        # Final 1x1 conv
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x2, x1 = self.encoder_bloc_1(x)
        x3, x2 = self.encoder_bloc_2(x2)
        x4, x3 = self.encoder_bloc_3(x3)
        x5, x4 = self.encoder_bloc_4(x4)

        # Bottleneck

        bn = self.bottle_neck(x5)

        # Decoder
        u1 = self.decoder_bloc_1(bn, x4)
        u2 = self.decoder_bloc_2(u1, x3)
        u3 = self.decoder_bloc_3(u2, x2)
        u4 = self.decoder_bloc_4(u3, x1)

        # Output
        return self.out(u4)