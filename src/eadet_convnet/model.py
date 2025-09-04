import numpy as np

import torch
from torch import nn
from torch.nn.functional import relu


def build_conv_block(
    channels, conv_kernels, conv_paddings, pool_kernels, pool_paddings
):
    """
    Parameters
    ----------
    channels : list of ints
        Conv2d channels. Number of channels (kernels) produced by the
        convolutions.
    conv_kernels : list of 2-tuple
        Conv2d kernels. Size of the convolving kernels.
    conv_paddings : list of 2-tuple
        Conv2d paddings. Zero-padding for convolution operations.
    pool_kernels : list of 2-tuple
        MaxPool2d kernel sizes. The size of the windows to take a max
        over.
    pool_paddings : list of 2-tuple
        Zero-padding for max-pooling operations.

    Returns
    -------
    conv_block : :py:class:`torch.nn.Sequential`
        Convolutional neural network.
    """
    nlayers = len(channels)
    channels = (1,) + tuple(channels)

    conv_block = nn.Sequential()
    for ilayer in range(nlayers):
        counter = ilayer + 1
        conv_block.add_module(
            "conv{}".format(counter),
            nn.Conv2d(
                in_channels=channels[ilayer],
                out_channels=channels[ilayer + 1],
                kernel_size=conv_kernels[ilayer],
                padding=conv_paddings[ilayer],
            ),
        )

        conv_block.add_module(
            "bnorm{}".format(counter), nn.BatchNorm2d(num_features=channels[ilayer + 1])
        )

        conv_block.add_module(
            "pool{}".format(counter),
            nn.MaxPool2d(
                kernel_size=pool_kernels[ilayer], padding=pool_paddings[ilayer]
            ),
        )

        conv_block.add_module("relu{}".format(counter), nn.ReLU())

    return conv_block


def calc_convlayer_outshape(conv2d_layer, in_shape):
    def f(d, k, s, p):
        return int(np.floor((d + (2 * p) - k) / s)) + 1

    in_h, in_w = in_shape
    kh, kw = conv2d_layer.kernel_size
    sh, sw = conv2d_layer.stride
    ph, pw = conv2d_layer.padding
    out_h = f(in_h, kh, sh, ph)
    out_w = f(in_w, kw, sw, pw)
    return (out_h, out_w)


def calc_convblock_outshape(conv_block, in_shape, nconv_layers):
    conv_inshape = in_shape
    for ilayer in range(nconv_layers):
        conv_outshape = calc_convlayer_outshape(
            conv_block._modules[f"conv{ilayer + 1}"], conv_inshape
        )

        pool_outshape = calc_convlayer_outshape(
            conv_block._modules[f"pool{ilayer + 1}"], conv_outshape
        )

        conv_inshape = pool_outshape

    return pool_outshape


class MultiTaskConvNet(nn.Module):
    def __init__(self, conv_block, in_shape, nconv_layers):
        super().__init__()
        self.conv_block = conv_block
        convblock_outshape = calc_convblock_outshape(conv_block, in_shape, nconv_layers)

        self.class_fc1 = nn.Linear(convblock_outshape, 512)
        self.class_fc2 = nn.Linear(512, 1)

        self.bbox_fc1 = nn.Linear(convblock_outshape, 512)
        self.bbox_fc2 = nn.Linear(512, 4)

    def forward(self, x_in):
        """
        Parameters
        ----------
        x_in : :py:class:`torch.Tensor`
            Batch of samples. It's a tensor of shape
            (N, C, H, W)=(batch_size, n_channels, n_stations, n_tpixels).

        Returns
        -------
        x_out : :py:class:`torch.Tensor`
            Batch of neural network output. It's a tensor of shape
            (N, P)=(batch_size, output_size), where `output_size` is
            the number of point-source parameters.
        """

        cnn_out = self.conv_block(x_in)
        cnn_out = cnn_out.view(-1, self.get_flatten_size(cnn_out))

        class_out = relu(self.class_fc1(cnn_out))
        class_out = self.class_fc2(class_out)

        bbox_out = relu(self.bbox_fc1(cnn_out))
        bbox_out = self.bbox_fc2(bbox_out)

        return (class_out, bbox_out)

    def get_flatten_size(self, a):
        return torch.prod(torch.tensor(a.shape[1:])).item()
