# -*- coding: utf-8 -*-
"""
@author: AmirPouya Hemmasian (ahemmasi@andrew.cmu.edu)
"""
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import mkdir, get_vp
np.seterr(divide='ignore', invalid='ignore')
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
room = 293
nt = 99


# %%
class GenClass():
    #  Initialization
    def __init__(self, name, data='Ti64-5', Tmax=6500):

        mkdir('Results/'+data)
        self.result_dir = 'Results/'+data+'/'+name
        self.outputs_dir = self.result_dir + '/outputs'
        mkdir(self.result_dir)
        mkdir(self.outputs_dir)
        self.prep_x = lambda x: torch.as_tensor(x/np.array([500, 1500, 100]), dtype=torch.float)
        self.prep_T = lambda T: (T-room)/(Tmax-room)
        self.post_T = lambda T: room + (Tmax-room)*T

    # %% Setting the Training and Validation Datasets
    def set_dataset(self, train_dataset, val_dataset):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    # %% Setting the main model, and its optimizer and learning curves
    def set_model(self, model, optimizer=optim.Adam, opt_kwargs={'lr':2e-4},
                  device=Device):

        self.model = model.to(device)
        self.optimizer = optimizer(self.model.parameters(), **opt_kwargs)
        self.Train_Loss = []
        self.Val_Loss = []

    # %% Setting the masker model and its optimizer and learninfg curves
    def set_masker(self, masker, optimizer=optim.Adam, opt_kwargs={'lr':2e-4},
                   device=Device):

        self.masker = masker.to(device)
        self.optimizer_masker = optimizer(self.masker.parameters(),
                                          **opt_kwargs)
        self.Train_Loss_masker = []
        self.Val_Loss_masker = []

    # %% Saving the state_dict of the main model, optimizer, learning curves
    def save_state_dict(self, path=None, masked=False):

        if path is None:
            path = self.result_dir+'/state_dict'
            if masked:
                path += '_masked'
        attrs = ['model', 'optimizer', 'Train_Loss', 'Val_Loss']
        state_dict = {attr: getattr(self, attr) for attr in attrs}
        with open(path+'.pickle', 'wb') as f:
            pickle.dump(state_dict, f)

    # %% Loading the state_dict
    def load_state_dict(self, path=None, masked=True):

        if path is None:
            path = self.result_dir + '/state_dict'
            if masked:
                path += '_masked'
        if masked:
            self.load_state_dict_masker()
        with open(path+'.pickle', 'rb') as f:
            state_dict = pickle.load(f)
        for key, value in state_dict.items():
            setattr(self, key, value)

    # %% Saving the state_dict for the masker model
    def save_state_dict_masker(self, path=None):

        if path is None:
            path = self.result_dir+'/state_dict_masker.pickle'

        attrs = ['masker', 'optimizer_masker',
                 'Train_Loss_masker', 'Val_Loss_masker']
        state_dict = {attr: getattr(self, attr) for attr in attrs}
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)

    # %% Loading the state_dict of the masker
    def load_state_dict_masker(self, path=None):

        if path is None:
            path = self.result_dir + '/state_dict_masker.pickle'
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
        for key, value in state_dict.items():
            setattr(self, key, value)

    # %% Training loop for the main model
    def train(self, num_epochs=50, batch_size=64, masked=False,
              criterion=nn.MSELoss(), device=Device):

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size,
                                shuffle=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                         factor=0.2,
                                                         patience=3)
        if masked:
            self.masker.requires_grad_(False).eval().to(device)
        self.model.to(device)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            ###################################################################
            # Training:
            losses = []
            self.model.requires_grad_().train()
            for x, T in tqdm(train_loader, position=0, leave=True):
                x = self.prep_x(x).to(device)
                T = self.prep_T(T).to(device)
                self.optimizer.zero_grad()
                T_pred = self.model(x)
                if masked:
                    mask = self.masker(x, mask=True) < 0.5
                    T_pred.masked_fill_(mask, 0)
                loss = criterion(T_pred, T)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                torch.cuda.empty_cache()
                del x, T, T_pred
            train_loss = np.mean(losses)
            self.Train_Loss.append(train_loss)
            scheduler.step(train_loss)
            ###################################################################
            # Validation:
            self.model.requires_grad_(False).eval()
            losses = []
            for x, T in tqdm(val_loader, position=0, leave=True):
                x = self.prep_x(x).to(device)
                T = self.prep_T(T).to(device)
                T_pred = self.model(x)
                if masked:
                    mask = self.masker(x, mask=True) < 0.5
                    T_pred.masked_fill_(mask, 0)
                loss = criterion(T_pred, T)
                losses.append(loss.item())
                torch.cuda.empty_cache()
                del x, T, T_pred
            val_loss = np.mean(losses)
            self.Val_Loss.append(val_loss)

            ###################################################################
            # Printing the results:
            print(f'Train Loss: {train_loss:.7f}  |  Val Loss: {val_loss:.7f}')
            print('~'*60)

            # self.save_state_dict(masked=masked)

    # %% Training loop for the masker model
    def train_masker(self, num_epochs=50, batch_size=64,
                     criterion=nn.MSELoss(), device=Device):

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(self.val_dataset,  batch_size=batch_size,
                                shuffle=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_masker,
                                                         factor=0.2,
                                                         patience=3)

        self.masker.to(device)
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            ###################################################################
            # Training:
            losses = []
            self.masker.requires_grad_().train()
            for x, T in tqdm(train_loader, position=0, leave=True):
                x = self.prep_x(x).to(device)
                T = self.prep_T(T).to(device)
                T = torch.as_tensor(T > 0, dtype=torch.float)
                self.optimizer_masker.zero_grad()
                T_pred = self.masker(x, mask=True)
                loss = criterion(T_pred, T)
                loss.backward()
                self.optimizer_masker.step()
                losses.append(loss.item())
                torch.cuda.empty_cache()
                del x, T, T_pred
            train_loss = np.mean(losses)
            self.Train_Loss_masker.append(train_loss)
            scheduler.step(train_loss)
            ###################################################################
            # Validation:
            self.masker.requires_grad_(False).eval()
            losses = []
            for x, T in tqdm(val_loader, position=0, leave=True):
                x = self.prep_x(x).to(device)
                T = self.prep_T(T).to(device)
                T = torch.as_tensor(T > 0, dtype=torch.float)
                T_pred = self.masker(x, mask=True)
                loss = criterion(T_pred, T)
                losses.append(loss.item())
                torch.cuda.empty_cache()
                del x, T, T_pred
            val_loss = np.mean(losses)
            self.Val_Loss_masker.append(val_loss)
            ###################################################################
            # Printing the results:
            print(f'Train Loss: {train_loss:.7f}  |  Val Loss: {val_loss:.7f}')
            print('~'*60)

            # self.save_state_dict_masker()

    # %% Setting the models to test mode
    def test_mode(self, device=Device):

        for model in ['model', 'masker']:
            try:
                getattr(self, model).requires_grad_(False).eval().to(device)
            except:
                pass

    # %% Get the output of the model for arbitrary (p, v, t)
    def test_sample(self, p, v, t, masked=True, device=Device):

        # remember to call self.test_mode(device) first
        x = self.prep_x([p, v, t]).unsqueeze(0).to(device)
        T_pred = self.model(x)
        if masked:
            mask = self.masker(x, mask=True) < 0.5
            T_pred.masked_fill_(mask, 0)
        T_pred = self.post_T(T_pred)
        return T_pred.squeeze().cpu().numpy()

    # %% Get the output of the model for a whole process with parameters p, v
    def test_process(self, p, v, masked=True, device=Device):

        Ts_pred = []
        for t in range(nt):
            x = self.prep_x([p, v, t]).unsqueeze(0).to(device)
            T_pred = self.model(x)
            if masked:
                mask = self.masker(x, mask=True) < 0.5
                T_pred.masked_fill_(mask, 0)
            Ts_pred.append(T_pred)
        Ts_pred = torch.cat(Ts_pred)
        Ts_pred = self.post_T(Ts_pred)
        return Ts_pred.squeeze().cpu().numpy()

    # %% validate the model on a sample from dataset
    def val_sample(self, sample, t=0, masked=True, device=Device):

        try:
            sample_idx = self.train_dataset.samples.index(sample)
            print('found in training data')
            x, T = self.train_dataset[sample_idx*nt+t]
        except:
            sample_idx = self.val_dataset.samples.index(sample)
            print('found in validation data')
            x, T = self.val_dataset[sample_idx*nt+t]
        v, p = get_vp(sample)
        T = T.squeeze().numpy()
        T_pred = self.test_sample(p, v, t, masked=masked, device=device)
        return (p, v, t), T, T_pred

    # %% Validation of the model on a sample from dataset
    def val_process(self, sample, masked=True, device=Device):

        v, p = get_vp(sample)
        Ts = []
        try:
            sample_idx = self.train_dataset.samples.index(sample)
            print('found in training data')
            for t in range(nt):
                T = self.train_dataset[sample_idx*nt+t][1]
                Ts.append(T.squeeze().numpy())
        except:
            sample_idx = self.val_dataset.samples.index(sample)
            print('found in validation data')
            for t in range(nt):
                T = self.val_dataset[sample_idx*nt+t][1]
                Ts.append(T.squeeze().numpy())
        Ts = np.stack(Ts)
        Ts_pred = self.test_process(p, v, masked=masked, device=device)
        return (p, v), Ts, Ts_pred

    # %% Validation of the model on a few samples
    def val_samples(self, sample, ts=range(10, 100, 20), masked=True,
                    device=Device):
        v, p = get_vp(sample)
        Ts = []
        Ts_pred = []
        try:
            sample_idx = self.train_dataset.samples.index(sample)
            print('found in training data')
            for t in ts:
                x, T = self.train_dataset[sample_idx*nt+t]
                Ts.append(T.squeeze().numpy())
                T_pred = self.test_sample(*x, masked=masked, device=device)
                Ts_pred.append(T_pred)
        except:
            sample_idx = self.val_dataset.samples.index(sample)
            print('found in validation data')
            for t in ts:
                x, T = self.val_dataset[sample_idx*nt+t]
                Ts.append(T.squeeze().numpy())
                T_pred = self.test_sample(*x, masked=masked, device=device)
                Ts_pred.append(T_pred)

        Ts = np.stack(Ts)
        Ts_pred = np.stack(Ts_pred)
        return (p, v, list(ts)), Ts, Ts_pred
