# -*- coding: utf-8 -*-
"""
@author: AmirPouya Hemmasian (ahemmasi@andrew.cmu.edu)
"""
import torch
from tqdm import tqdm
from utils import load, remake, get_pv
room = 293
nt = 99


class AM_3D_Dataset(torch.utils.data.Dataset):

    def __init__(self, samples, data='Ti64-5', data_type='temperature'):

        super().__init__()
        self.num_samples = len(samples)
        self.samples = list(samples)
        self.remake = lambda cropped, slices: remake(cropped, slices, data)

        print('Loading Dataset ...')
        self.dataset = []
        for sample in tqdm(samples):
            self.dataset.append([
                load(sample, data, t, data_type)
                for t in range(1, nt+1)
                ])

        self.hashmap = lambda i: (i//nt, i%nt)

    def __len__(self):
        return self.num_samples*nt

    def __getitem__(self, index):
        n, t = self.hashmap(index)
        T = self.remake(*self.dataset[n][t])
        T = torch.as_tensor(T, dtype=torch.float).unsqueeze(0)
        p, v = get_pv(self.samples[n])
        x = torch.tensor([p, v, t], dtype=torch.float)
        return x, T
