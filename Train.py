# -*- coding: utf-8 -*-
"""
@author: AmirPouya Hemmasian (ahemmasi@andrew.cmu.edu)
"""
import torch
from copy import deepcopy
from utils import get_samples, model_shapes, model_slices, get_train_val
from Dataset import AM_3D_Dataset
from GenClass import GenClass
from Model import Generator3d
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_idx = int(input('Choose the dataset: (1) Ti64-5 | (2) Ti64-10 | (3) Ti64-10-p : '))
datas = {1: 'Ti64-5', 2:'Ti64-10', 3:'Ti64-10-p'}
data = datas[data_idx]

num_epochs = int(input('Choose the number of epochs : '))

up_idx = int(input('Choose layer type: (1) trilinear | (2) 4x4 upconv | (3) 2x2 upconv : '))

up_types = {
    1: dict(mode='trilinear', scale_factor=2, align_corners=False),
    2: dict(kernel_size=4, stride=2, padding=1),
    3: dict(kernel_size=2, stride=2, padding=0)
    }

def get_model_name(kwargs):
    name = ''
    name += f"FC{kwargs.get('fc_hidden', [])}"
    name += f"_CNN{kwargs.get('channels', [])}"
    up_kwargs = kwargs.get('up_kwargs', [])
    try:
        k = up_kwargs['kernel_size']
        name += f'_tconv{k}x{k}'
    except:
        pass
    return name

kwargs = dict(# do not change these two arguments
              shape=model_shapes[data],
              slices=model_slices[data],
              ###############################################################
              fc_hiddens=[], channels=[128, 64, 32, 16, 8, 4],
              up_kwargs=up_types[up_idx]
              )

samples = get_samples(data)
train, val = get_train_val(samples, data)

name = get_model_name(kwargs)

Model = GenClass(name, data)

Model.set_dataset(
    AM_3D_Dataset(samples[train], data),
    AM_3D_Dataset(samples[val], data)
    )

# Training T-CNN
Model.set_model(Generator3d(**kwargs))
Model.train(num_epochs, masked=False)

# Training M-CNN
Model.set_masker(deepcopy(Model.model))  # transfer learning
Model.train_masker(num_epochs)

# Training MT-CNN
Model.set_model(Generator3d(**kwargs))
Model.train(num_epochs, masked=True)
