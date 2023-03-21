# -*- coding: utf-8 -*-
"""
@author: AmirPouya Hemmasian (ahemmasi@andrew.cmu.edu)
"""

import numpy as np
import os
import torch
import itertools
import matplotlib.pyplot as plt
from matplotlib import animation
# matplotlib.rcParams['figure.autolayout'] = True

# Helper function to make a new directory
mkdir = lambda path: os.mkdir(path) if not os.path.exists(path) else None


def what_is(x):
    print('type:', type(x))
    try:
        print('shape:', x.shape)
    except:
        pass
# %% pre-defined parameters (do not change)
room = 293
nt = 99

roi_shapes = {
        'Ti64-5': (116, 32, 64),  # out of (200,120,80)
        'Ti64-10': (60, 20, 40),  # out of (100,60,40)
        'Ti64-10-p': (84, 32, 19)  # out of (100,60,25)
}

model_shapes = {
        'Ti64-5': (128, 32, 64),
        'Ti64-10': (64, 32, 64),
        'Ti64-10-p': (128, 32, 32)
}

model_slices = {
        'Ti64-5': [slice(6, -6), slice(None, None), slice(None, None)],
        'Ti64-10': [slice(2, -2), slice(6, -6), slice(12, -12)],
        'Ti64-10-p': [slice(22, -22), slice(None, None), slice(6, -7)]
}

vals = {'Ti64-5': [3, 14, 16, 24, 28, 34, 44]}


def get_vp(x):
    x = x.split('_')
    return int(x[-1]), int(x[-3])

def get_pv(x):
    x = x.split('_')
    return int(x[-3]), int(x[-1])

def get_meshes(data, cropped=True):
    dr = 'Datasets/'+data
    if cropped: dr += '_cropped'
    mesh_x = np.load(dr+'/mesh_x.npy', allow_pickle=True)
    mesh_y = np.load(dr+'/mesh_y.npy', allow_pickle=True)
    mesh_z = np.load(dr+'/mesh_z.npy', allow_pickle=True)
    return mesh_x, mesh_y, mesh_z


def get_samples(data, cropped=True):
    dr = 'Datasets/'+data
    if cropped: dr += '_cropped'
    samples = sorted([s for s in os.listdir(dr) if s.startswith('melt')],
                     key=get_pv)
    return np.array(samples)


def get_train_val(samples, data, seed=0):
    np.random.seed(seed) # for reproducability of the results
    N = len(samples)
    if data=='Ti64-5':
        val_idxs = [3, 14, 16, 24, 28, 34, 44]
    else:
        val_idxs = np.random.choice(N, N//5, False)
    train_idxs = set(range(N))-set(val_idxs)
    return list(train_idxs), list(val_idxs)


def viz_dataset(samples, val=[], annotate=0, s=100, save=''):

    N = len(samples)
    vps = np.array([get_vp(sample) for sample in samples])
    train = list(set(range(N))-set(val))
    plt.figure(figsize=(8, 6))
    if annotate:
        for i, (v, p) in enumerate(vps):
            plt.annotate(i, (v+2, p+2), fontsize=annotate)
    plt.scatter(*zip(*vps[train]), label='Train', s=s)
    if len(val):
        plt.scatter(*zip(*vps[val]), label='Val', s=s)
    plt.grid(linestyle='--')
    plt.xlabel('Velocity (mm/s)', fontsize=14)
    plt.ylabel('Power (w)', fontsize=14)
    plt.legend(loc=(1.01, 0.85), fontsize=14)
    plt.title('Dataset', fontsize=14)
    if save:
        plt.savefig(save+'/dataset.jpg', bbox_inches='tight')
    plt.show()
    print(f'total: {N} | train: {len(train)} | val: {len(val)}')


# %%
def load(sample, data='Ti64-5', t=0, data_type='temperature', cropped=True):
    """
    loads a single data file as a numpy array of dtype float32
    sample:
        the folder name of the corresponding sample
    data:
        'Ti64-5' or 'Ti64-10' or 'Ti64-10-p'
    t:
        timestep (int in range(1,100))
    data_type:
        one of these:
        'temperature', 'meltregion', 'liqlabel', 'mesh_x', 'mesh_y', 'mesh_z'
    cropped:
        whether to load the preprocessed (cropped) data
    """
    dr = 'Datasets/' + data
    if cropped: dr += '_cropped'
    dr += '/' + sample + '/' + data_type

    if not data_type.startswith('mesh') and t:
        dr += f'_{t:02d}'

        if cropped:
            dr1 = dr + '_cropped.npy'
            dr2 = dr + '_slice.npy'

            cropped = np.load(dr1, allow_pickle=True)
            slices = np.load(dr2, allow_pickle=True)
            return cropped, slices

    x = np.load(dr+'.npy', allow_pickle=True).astype(np.float32)
    if data_type.startswith('mesh'): x.sort()
    return x


# %% Data processing function:
def remake(cropped, slices, data, fill=room):
    X = np.full(roi_shapes[data], fill, dtype=np.float32)
    (xs, xe), (ys, ye), (zs, ze) = slices
    X[xs:xe, ys:ye, zs:ze] = cropped
    return X


# %% Helper function to extract 2D cross section from 3D field
def cross_section(T, rx=None, ry=None, rz=None):
    """
    returns a cross section of T
    specify either x, y or z (float between 0 and 1)
    """
    nx, ny, nz = np.array(T.shape) - 1
    if sum([rx is None, ry is None, rz is None]) != 2:
        print('Specify either x, y, z in [0,1]')
        return
    if rx is not None:
        return T[round(rx*nx), :, :].T
    if ry is not None:
        return T[:, round(ry*ny), :].T
    if rz is not None:
        return T[:, :, round(rz*nz)].T


# %% Function for visualizing a sample
def viz_sample(T, data, ry=0.5, rz=1, cmap='jet', melt=1873,
               shading='gouraud', figsize=(6, 8)):

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
    mx, my, mz = get_meshes(data)
    idy = np.interp(ry, np.linspace(0, 1, len(my)), my)
    idz = np.interp(rz, np.linspace(0, 1, len(mz)), mz)
    ax[0].pcolormesh(mx, my, cross_section(T, rz=rz),shading=shading,
                     cmap=cmap, vmin=room, vmax=melt)
    im01 = ax[1].pcolormesh(mx, mz, cross_section(T, ry=ry), shading=shading,
                            cmap=cmap, vmin=room, vmax=melt)

    ax[0].set_ylabel(f'y [cm] (z={idz:.3f})', fontsize=14)
    ax[1].set_ylabel(f'z [cm] (y={idy:.3f})', fontsize=14)
    ax[1].set_xlabel('x [cm]', fontsize=14)

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    ax[0].hlines(idy, mx[0], mx[-1], color='green', linewidth=1)
    ax[1].hlines(idz, mx[0], mx[-1], color='green', linewidth=1)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.12, 0.01, 0.75])
    fig.colorbar(im01, cax=cbar_ax)
    cbar_ax.set_title('T(K)')
    plt.show()


# %% Function for visualizing a process
def viz_process(Ts, data, ry=0.5, rz=1, cmap='jet', melt=1873,
                shading='gouraud', figsize=(6, 8)):

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
    mx, my, mz = get_meshes(data)
    idy = np.interp(ry, np.linspace(0, 1, len(my)), my)
    idz = np.interp(rz, np.linspace(0, 1, len(mz)), mz)
    im0 = ax[0].pcolormesh(mx, my, cross_section(Ts[0], rz=rz),
                               shading=shading, cmap=cmap,
                               vmin=room, vmax=melt)
    im1 = ax[1].pcolormesh(mx, mz, cross_section(Ts[0], ry=ry),
                               shading=shading, cmap=cmap,
                               vmin=room, vmax=melt)

    ax[0].set_ylabel(f'y [cm] (z={idz:.3f})', fontsize=14)
    ax[1].set_ylabel(f'z [cm] (y={idy:.3f})', fontsize=14)
    ax[1].set_xlabel('x [cm]', fontsize=14)

    ax[0].hlines(idy, mx[0], mx[-1], color='green', linewidth=1)
    ax[1].hlines(idz, mx[0], mx[-1], color='green', linewidth=1)

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.12, 0.01, 0.75])
    fig.colorbar(im1, cax=cbar_ax)
    cbar_ax.set_title('T(K)')

    def animate(i):
        im0.set_array(cross_section(Ts[i], rz=rz))
        im1.set_array(cross_section(Ts[i], ry=ry))
        ax[0].set_title(f't = {i}', fontsize=16)

    ani = animation.FuncAnimation(fig, animate, frames=nt,
                                  interval=100, repeat=True)

    plt.show()
    return ani


# %% Helper function to get important temperature range:
def get_solidifying_region(Ts, melt=1873, start_temp=1229):
    idx1 = np.logical_and(start_temp < Ts, Ts < melt)
    idx2 = np.logical_and(room < Ts, Ts < melt)
    idx2 = idx2[::-1].cumsum(0)[::-1] == np.arange(1, nt+1)[::-1, None, None, None]
    idx = np.logical_and(idx1, idx2)
    Ts_sol = Ts.copy()
    Ts_sol[~idx] = np.nan
    return Ts_sol


# %% Functions for vizual comparison
def compare_sample(x, Ts, Ts_pred, data,
                   Tevap=2500, melt=1873, start_temp=1229,
                   erange=500, ry=0.5, rz=0.9, cmap='jet',
                   save='', shading='gouraud', figsize=(15, 6)):

    p, v, t = x
    T = Ts[t]
    T_pred = Ts_pred[t]
    error = Ts_pred - get_solidifying_region(Ts, melt, start_temp)
    rmse = np.nanmean(error**2, axis=(1, 2, 3))**0.5
    mx, my, mz = get_meshes(data)
    idy = np.interp(ry, np.linspace(0, 1, len(my)), my)
    idz = np.interp(rz, np.linspace(0, 1, len(mz)), mz)
    vmin, vmax = room, melt

    fig, ax = plt.subplots(2, 3, figsize=figsize, sharex=True)

    ax[0, 0].pcolormesh(mx, my, cross_section(T, rz=rz),
                               shading=shading, cmap=cmap,
                               vmin=vmin, vmax=vmax)
    ax[1, 0].pcolormesh(mx, mz, cross_section(T, ry=ry),
                               shading=shading, cmap=cmap,
                               vmin=vmin, vmax=vmax)
    ax[0, 1].pcolormesh(mx, my, cross_section(T_pred, rz=rz),
                               shading=shading, cmap=cmap,
                               vmin=vmin, vmax=vmax)
    im11 = ax[1, 1].pcolormesh(mx, mz, cross_section(T_pred, ry=ry),
                               shading=shading, cmap=cmap,
                               vmin=vmin, vmax=vmax)
    ax[0, 2].pcolormesh(mx, my, cross_section(error[t], rz=rz),
                               shading=shading, cmap='seismic',
                               vmin=-erange, vmax=erange)
    im12 = ax[1, 2].pcolormesh(mx, mz, cross_section(error[t], ry=ry),
                               shading=shading, cmap='seismic',
                               vmin=-erange, vmax=erange)

    ax[0, 0].hlines(idy, mx[0], mx[-1], color='green', linewidth=1)
    ax[1, 0].hlines(idz, mx[0], mx[-1], color='green', linewidth=1)
    ax[0, 1].hlines(idy, mx[0], mx[-1], color='green', linewidth=1)
    ax[1, 1].hlines(idz, mx[0], mx[-1], color='green', linewidth=1)
    ax[0, 2].hlines(idy, mx[0], mx[-1], color='green', linewidth=1)
    ax[1, 2].hlines(idz, mx[0], mx[-1], color='green', linewidth=1)

    ax[0, 0].set_title('FLOW-3D', fontsize=16)
    ax[0, 1].set_title('CNN', fontsize=16)
    ax[0, 2].set_title('Error', fontsize=16)

    ax[1, 0].set_xlabel('x [cm]', fontsize=12)
    ax[1, 1].set_xlabel('x [cm]', fontsize=12)
    ax[1, 2].set_xlabel('x [cm]', fontsize=12)

    ax[0, 0].set_ylabel(f'y [cm] (z={idz:.3f})', fontsize=12)
    ax[1, 0].set_ylabel(f'z [cm] (y={idy:.3f})', fontsize=12)

    ax[1, 0].set_title(f'P = {p} w, V = {v} mm/s', fontsize=12)

    ax[1, 1].set_title(f't = {t}', fontsize=14)
    ax[1, 2].set_title(f'RMSE = {rmse[t]:.1f} K ({100*rmse[t]/(Tevap-room):.2f}%)', fontsize=14)

    for i, j in itertools.product(range(2), range(3)):
        ax[i, j].set_aspect('equal')

    fig.subplots_adjust(right=0.9, left=0.1)
    cbar_ax = fig.add_axes([0.0, 0.12, 0.01, 0.75])
    fig.colorbar(im11, cax=cbar_ax)
    cbar_ax.set_title('    T(K)')

    cbar_ax2 = fig.add_axes([0.92, 0.12, 0.01, 0.75])
    fig.colorbar(im12, cax=cbar_ax2)
    cbar_ax2.set_title('T(K)')


    if save:
        plt.savefig(save+'.jpg', dpi=600, bbox_inches='tight')
    plt.show()


def compare_process(x, Ts, Ts_pred, data, ry=0.5, rz=0.9,
                    Tevap=2500, melt=1873, erange=500, start_temp=1229,
                    cmap='jet', save='', shading='gouraud', figsize=(15, 6)):

    p, v = x
    mx, my, mz = get_meshes(data)
    idy = np.interp(ry, np.linspace(0, 1, len(my)), my)
    idz = np.interp(rz, np.linspace(0, 1, len(mz)), mz)
    error = Ts_pred - get_solidifying_region(Ts, melt, start_temp)
    rmse = np.nanmean(error**2, axis=(1, 2, 3))**0.5
    label = lambda i: f'RMSE = {rmse[i]:.1f} K ({100*rmse[i]/(Tevap-room):.2f}%)'
    vmin, vmax = room, melt

    fig, ax = plt.subplots(2, 3, figsize=figsize, sharex=True)

    im00 = ax[0, 0].pcolormesh(mx, my, cross_section(Ts[0], rz=rz),
                               shading=shading, cmap=cmap,
                               vmin=vmin, vmax=vmax)
    im10 = ax[1, 0].pcolormesh(mx, mz, cross_section(Ts[0], ry=ry),
                               shading=shading, cmap=cmap,
                               vmin=vmin, vmax=vmax)
    im01 = ax[0, 1].pcolormesh(mx, my, cross_section(Ts_pred[0], rz=rz),
                               shading=shading, cmap=cmap,
                               vmin=vmin, vmax=vmax)
    im11 = ax[1, 1].pcolormesh(mx, mz, cross_section(Ts_pred[0], ry=ry),
                               shading=shading, cmap=cmap,
                               vmin=vmin, vmax=vmax)
    im02 = ax[0, 2].pcolormesh(mx, my, cross_section(error[0], rz=rz),
                               shading=shading, cmap='seismic',
                               vmin=-erange, vmax=erange)
    im12 = ax[1, 2].pcolormesh(mx, mz, cross_section(error[0], ry=ry),
                               shading=shading, cmap='seismic',
                               vmin=-erange, vmax=erange)

    ax[0, 0].hlines(idy, mx[0], mx[-1], color='green', linewidth=1)
    ax[1, 0].hlines(idz, mx[0], mx[-1], color='green', linewidth=1)
    ax[0, 1].hlines(idy, mx[0], mx[-1], color='green', linewidth=1)
    ax[1, 1].hlines(idz, mx[0], mx[-1], color='green', linewidth=1)
    ax[0, 2].hlines(idy, mx[0], mx[-1], color='green', linewidth=1)
    ax[1, 2].hlines(idz, mx[0], mx[-1], color='green', linewidth=1)

    ax[0, 0].set_title('FLOW-3D', fontsize=16)
    ax[0, 1].set_title('CNN', fontsize=16)
    ax[0, 2].set_title('Error', fontsize=16)

    ax[1, 0].set_xlabel('x [cm]', fontsize=12)
    ax[1, 1].set_xlabel('x [cm]', fontsize=12)
    ax[1, 2].set_xlabel('x [cm]', fontsize=12)

    ax[0, 0].set_ylabel(f'y [cm] (z={idz:.3f})', fontsize=12)
    ax[1, 0].set_ylabel(f'z [cm] (y={idy:.3f})', fontsize=12)

    ax[1, 0].set_title(f'P = {p} w, V = {v} mm/s', fontsize=12)

    for i, j in itertools.product(range(2), range(3)):
        ax[i, j].set_aspect('equal')

    fig.subplots_adjust(right=0.9, left=0.1)
    cbar_ax = fig.add_axes([0.0, 0.12, 0.01, 0.75])
    fig.colorbar(im11, cax=cbar_ax)
    cbar_ax.set_title('    T(K)')

    cbar_ax2 = fig.add_axes([0.92, 0.12, 0.01, 0.75])
    fig.colorbar(im12, cax=cbar_ax2)
    cbar_ax2.set_title('T(K)')

    def animate(i):
        im00.set_array(cross_section(Ts[i], rz=rz))
        im10.set_array(cross_section(Ts[i], ry=ry))
        im01.set_array(cross_section(Ts_pred[i], rz=rz))
        im11.set_array(cross_section(Ts_pred[i], ry=ry))
        im02.set_array(cross_section(error[i], rz=rz))
        im12.set_array(cross_section(error[i], ry=ry))
        ax[1, 1].set_title(f't = {i}', fontsize=14)
        ax[1, 2].set_title(label(i), fontsize=14)

    ani = animation.FuncAnimation(fig, animate, frames=nt,
                                  interval=100, repeat=True)
    if save:
        writer = animation.FFMpegWriter(fps=10)
        ani.save(save+'.mp4', writer=writer)
    plt.show()
    return ani

pv_xys = {
    'Ti64-5': (0.058, 0.025),
    'Ti64-10': (0.058, 0.02),
    'Ti64-10-p': (0, 0.01),
    'SS316L': (0.072, 0.01)
}

error_xys = {
    'Ti64-5': (0.058, -0.01),
    'Ti64-10': (0.058, -0.022),
    'Ti64-10-p': (0, -0.01),
    'SS316L': (0.072, -0.025)
}


def compare_samples(x, Ts, Ts_pred, data, save='', ts=range(10, 100, 20),
                    melt=1873, Tevap=2500, start_temp=1229,
                    ry=0.5, cmap='jet', show_stats=True, shading='gouraud'):

    p, v = x
    mx, my, mz = get_meshes(data)
    n = len(ts)
    error = Ts_pred - get_solidifying_region(Ts, melt, start_temp)
    rmse = np.nanmean(error**2, axis=(1, 2, 3))**0.5

    Ms = Ts > melt
    Ms_pred = Ts_pred > melt
    I = np.logical_and(Ms, Ms_pred).sum((1, 2, 3))
    U = np.logical_or(Ms, Ms_pred).sum((1, 2, 3))
    IoU = I/U

    fig, ax = plt.subplots(2, n, figsize=(2.5*n, 2.5),
                           sharex=True, sharey=True)

    #ax[1, 0].text(*pv_xys[data], f'P = {p} w\nV = {v} mm/s',
                  #fontsize=18, rotation=90, ma='center')
    if show_stats:
        ax[1, 0].text(*error_xys[data], 'RMSE:\nIoU:',
                      fontsize=16, ma='center')

    for i, t in enumerate(ts):
        ax[0, i].pcolormesh(mx, mz, cross_section(Ts[t], ry=ry),
                                  shading=shading, cmap=cmap,
                                  vmin=room, vmax=melt)
        im1 = ax[1, i].pcolormesh(mx, mz, cross_section(Ts_pred[t], ry=ry),
                                  shading=shading, cmap=cmap,
                                  vmin=room, vmax=melt)
        ax[0, i].set_title(f't = {t}', fontsize=18)
        ax[0, i].get_xaxis().set_ticks([])
        ax[1, i].get_yaxis().set_ticks([])

        if show_stats:
            stat = f'{rmse[t]:.0f} K ({100*rmse[t]/(Tevap-room):.1f}%)'
            stat += f'\n{100*IoU[t]:.1f}%'
            ax[1, i].set_xlabel(stat, fontsize=16)

    ax[0, 0].set_ylabel('FLOW-3D', fontsize=14, loc='center')
    ax[1, 0].set_ylabel('CNN', fontsize=14, loc='center')


    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.97, 0.12, 0.01, 0.77])
    fig.colorbar(im1, cax=cbar_ax)
    cbar_ax.set_title('T(K)', fontsize=16)
    cbar_ax.tick_params(labelsize=14)
    cbar_ax.get_yaxis().set_ticks([600, 1000, 1400, 1800])

    if save:
        plt.savefig(save+'.jpg', bbox_inches='tight')
    plt.show()

def compare_samples_z(x, Ts, Ts_pred, data, save='', ts=range(10, 100, 20),
                    melt=1873, Tevap=2500, start_temp=1229,
                    rz=0.9, cmap='jet', show_stats=True, shading='gouraud'):

    p, v = x
    mx, my, mz = get_meshes(data)
    n = len(ts)
    error = Ts_pred - get_solidifying_region(Ts, melt, start_temp)
    rmse = np.nanmean(error**2, axis=(1, 2, 3))**0.5

    Ms = Ts > melt
    Ms_pred = Ts_pred > melt
    I = np.logical_and(Ms, Ms_pred).sum((1, 2, 3))
    U = np.logical_or(Ms, Ms_pred).sum((1, 2, 3))
    IoU = I/U

    fig, ax = plt.subplots(2, n, figsize=(2.5*n, 2.5),
                           sharex=True, sharey=True)

    ax[1, 0].text(0.058, 0.028, f'P = {p} w\nV = {v} mm/s',
                  fontsize=18, rotation=90, ma='center')
    if show_stats:
        ax[1, 0].text(*error_xys[data], 'RMSE:\nIoU:',
                      fontsize=16, ma='center')

    for i, t in enumerate(ts):
        ax[0, i].pcolormesh(mx, my, cross_section(Ts[t], rz=rz),
                                  shading=shading, cmap=cmap,
                                  vmin=room, vmax=melt)
        im1 = ax[1, i].pcolormesh(mx, my, cross_section(Ts_pred[t], rz=rz),
                                  shading=shading, cmap=cmap,
                                  vmin=room, vmax=melt)
        ax[0, i].set_title(f't = {t}', fontsize=18)
        if show_stats:
            stat = f'{rmse[t]:.0f} K ({100*rmse[t]/(Tevap-room):.1f}%)'
            stat += f'\n{100*IoU[t]:.1f}%'
            ax[1, i].set_xlabel(stat, fontsize=16)

    ax[0, 0].set_ylabel('FLOW-3D', fontsize=14, loc='center')
    ax[1, 0].set_ylabel('CNN', fontsize=14, loc='center')

    for i, j in itertools.product(range(2), range(n)):
        ax[i, j].get_xaxis().set_ticks([])
        ax[i, j].get_yaxis().set_ticks([])

    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.97, 0.12, 0.01, 0.77])
    fig.colorbar(im1, cax=cbar_ax)
    cbar_ax.set_title('T(K)', fontsize=16)
    cbar_ax.tick_params(labelsize=14)
    cbar_ax.get_yaxis().set_ticks([600, 1000, 1400, 1800])

    if save:
        plt.savefig(save+'.jpg', bbox_inches='tight')
    plt.show()
    plt.close()

# %% Helper functions to get RMSE and IoU over the whole dataset and plot them
def get_rmse_iou(Ts, Ts_pred, melt=1873, start_temp=1229):

    error = Ts_pred - get_solidifying_region(Ts, melt, start_temp)
    rmse = np.nanmean(error**2, axis=(1, 2, 3))**0.5
    Ms = Ts > melt
    Ms_pred = Ts_pred > melt
    I = np.logical_and(Ms, Ms_pred).sum((1, 2, 3))
    U = np.logical_or(Ms, Ms_pred).sum((1, 2, 3))
    IoU = I/U
    return rmse, IoU


def plot_result_vs_t(RMSE, IoU, train, val, Tevap=2500, alpha=0.15,
                     rmse_max=10, figsize=(16, 5), save='result_t'):

    plt.figure(figsize=figsize)
    xaxis = range(nt)
    plt.subplot(121)
    line = np.nanmedian(RMSE[train, :], 0)*100/(Tevap-room)
    shade_bottom = np.nanpercentile(RMSE[train, :], 25, 0)*100/(Tevap-room)
    shade_top = np.nanpercentile(RMSE[train, :], 75, 0)*100/(Tevap-room)
    plt.plot(xaxis, line, color='red', linewidth=3)
    plt.fill_between(xaxis, shade_bottom, shade_top,
                     alpha=alpha, color='red')
    plt.grid(linestyle='--')
    plt.xlim([-2, 100])
    plt.ylim([0, rmse_max])
    plt.xticks(range(0, 100, 10), fontsize=16)
    plt.yticks(np.linspace(0, rmse_max, 11), fontsize=16)
    plt.xlabel('time step', fontsize=18)
    plt.ylabel('Relative RMSE (%)', color='red', fontsize=18)

    plt.gca().twinx()
    line = np.nanmedian(IoU[train, :], 0)*100
    shade_bottom = np.nanpercentile(IoU[train, :], 25, 0)*100
    shade_top = np.nanpercentile(IoU[train, :], 75, 0)*100

    plt.plot(xaxis, line, color='blue', linewidth=3)
    plt.fill_between(xaxis, shade_bottom, shade_top,
                     alpha=alpha, color='blue')
    plt.ylim([0, 100])
    plt.yticks(range(0, 101, 10), fontsize=16)
    plt.ylabel('Melt Pool IoU (%)', fontsize=18, color='blue')
    plt.title('Train Dataset', fontsize=20)

    plt.subplot(122)
    line = np.nanmedian(RMSE[val, :], 0)*100/(Tevap-room)
    shade_bottom = np.nanpercentile(RMSE[val, :], 25, 0)*100/(Tevap-room)
    shade_top = np.nanpercentile(RMSE[val, :], 75, 0)*100/(Tevap-room)
    plt.plot(xaxis, line, color='red', linewidth=3)
    plt.fill_between(xaxis, shade_bottom, shade_top,
                     alpha=alpha, color='red')
    plt.grid(linestyle='--')
    plt.xlim([-2, 100])
    plt.ylim([0, rmse_max])
    plt.xticks(range(0, 100, 10), fontsize=16)
    plt.yticks(np.linspace(0, rmse_max, 11), fontsize=16)
    plt.xlabel('time step', fontsize=18)
    plt.ylabel('Relative RMSE (%)', color='red', fontsize=18)

    plt.gca().twinx()
    line = np.nanmedian(IoU[val, :], 0)*100
    shade_bottom = np.nanpercentile(IoU[val, :], 25, 0)*100
    shade_top = np.nanpercentile(IoU[val, :], 75, 0)*100
    plt.plot(xaxis, line, color='blue', linewidth=3)
    plt.fill_between(xaxis, shade_bottom, shade_top,
                     alpha=alpha, color='blue')
    plt.ylim([0, 100])
    plt.yticks(range(0, 101, 10), fontsize=16)
    plt.ylabel('Melt Pool IoU (%)', fontsize=18, color='blue')
    plt.title('Validation Dataset', fontsize=20)

    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.show()


def plot_result_vs_PV(RMSE, IoU, samples, train, val, s=70, rmse_vmax=5,
                      figsize=(16, 5), Tevap=2500, save='result_PV', edge=2):

    N = len(samples)
    vps = np.array([get_vp(sample) for sample in samples])
    plt.figure(figsize=figsize)
    plt.subplot(121)
    edgecolors = np.array(['white']*N)
    edgecolors[val] = 'black'
    ax = plt.scatter(*zip(*vps), c=np.nanmean(RMSE, 1)/(Tevap-room)*100,
                     cmap='jet', s=s, edgecolors=edgecolors,
                     vmin=0, vmax=rmse_vmax, linewidth=edge)
    plt.grid(linestyle='--')

    cb = plt.colorbar(ax)
    cb.ax.tick_params(labelsize=16)
    plt.xlabel('Velocity (mm/s)', fontsize=18)
    plt.ylabel('Power (w)', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Average Relative RMSE(%)', fontsize=20)

    plt.subplot(122)
    ax = plt.scatter(*zip(*vps), c=np.nanmean(IoU, 1)*100,
                     cmap='jet', s=s, edgecolors=edgecolors,
                     vmin=50, vmax=100, linewidth=edge)
    plt.grid(linestyle='--')
    cb = plt.colorbar(ax)
    cb.ax.tick_params(labelsize=16)
    plt.xlabel('Velocity (mm/s)', fontsize=18)
    plt.ylabel('Power (w)', fontsize=18)
    plt.title('Average Melt Pool IoU (%)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.show()
