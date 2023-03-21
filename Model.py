# -*- coding: utf-8 -*-
"""
@author: AmirPouya Hemmasian (ahemmasi@andrew.cmu.edu)
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


def upsample(c_in, c_out, up_kwargs, act=nn.LeakyReLU):

    if 'mode' in up_kwargs:
        layers = [
            nn.Upsample(**up_kwargs),
            nn.Conv3d(c_in, c_out, 3, padding='same', padding_mode='replicate')
        ]
    else:
        layers = [nn.ConvTranspose3d(c_in, c_out, **up_kwargs)]

    layers += [act()]
    return nn.Sequential(*layers)


class Generator3d(nn.Module):

    def __init__(self, shape=(128, 32, 64),
                 fc_hiddens=[],
                 channels=[128, 64, 32, 16, 8, 4],
                 up_kwargs=dict(scale_factor=2, mode='trilinear',
                                align_corners=False),
                 act=nn.LeakyReLU,
                 slices=3*[slice(None, None)],
                 ):

        super().__init__()
        self.slice_x, self.slice_y, self.slice_z = slices
        #######################################################################
        cd = 2**(len(channels)-1)
        c1 = channels[0] if channels else 1
        d1, d2, d3 = shape
        self.start_shape = (c1, d1//cd, d2//cd, d3//cd)

        fc_layers = []
        fc_units = [3] + fc_hiddens + [np.prod(self.start_shape)]
        for i in range(len(fc_units)-1):
            fc_layers += [nn.Linear(fc_units[i], fc_units[i+1]), act()]
        self.fcn = nn.Sequential(*fc_layers)
        #######################################################################
        conv_layers = []
        for i in range(len(channels)-1):
            conv_layers += [
                upsample(channels[i], channels[i+1], up_kwargs, act)
                ]
        conv_layers += [nn.Conv3d(channels[-1], 1, 3, padding='same',
                                  padding_mode='replicate')]
        self.cnn = nn.Sequential(*conv_layers)

    def forward(self, u, mask=False):
        x = self.fcn(u).reshape(u.shape[0], *self.start_shape)
        x = self.cnn(x)
        x = x[:, :, self.slice_x, self.slice_y, self.slice_z]
        if mask:
            return torch.sigmoid(x)
        return F.leaky_relu(x, 0.001 if self.training else 0)
