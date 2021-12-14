import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

import os
import pickle

plt.rcParams.update({'font.size': 32})
plt.rcParams.update({'image.interpolation': None})
plt.rcParams['figure.figsize'] = [14, 12]
plt.rcParams['figure.dpi'] = 200


#%%
load_dir = '/Users/elizabethnewman/Desktop/hessQuikResults/scalar_output/'

os.chdir(load_dir)

markers = ['o', '^', 's']
linewidth = 6
markersize = 20


plt.figure()
for i, name in enumerate(['hessQuik', 'PytorchAD', 'PytorchHessian']):
    output = pickle.load(open('12-13-2021---' + name + '-cpu-w20-d4.p', 'rb'))
    results = output['results']

    x = results['in_feature_range']
    y = results['timing_trials_mean'].squeeze()

    plt.semilogy(x, y, '-' + markers[i], linewidth=linewidth, markersize=markersize, label=name + ': cpu')

    output = pickle.load(open('12-13-2021---' + name + '-cuda-w20-d4.p', 'rb'))
    results = output['results']

    x = results['in_feature_range']
    y = results['timing_trials_mean'].squeeze()

    plt.semilogy(x, y, '--' + markers[i], linewidth=linewidth, markersize=markersize, label=name + ': cuda')

plt.xscale('log', base=2)
plt.yscale('log', base=10)
plt.xlabel('in features')
plt.ylabel('time (seconds)')

plt.legend()
plt.show()

#%%
load_dir = '/Users/elizabethnewman/Desktop/hessQuikResults/vector_output/'

os.chdir(load_dir)

plt.figure()
for i, name in enumerate(['hessQuik', 'PytorchAD']):
    output = pickle.load(open('12-13-2021---' + name + '-cpu-w20-d4.p', 'rb'))
    results = output['results']
    timing_trials_mean = results['timing_trials_mean']
    in_feature_range = results['in_feature_range']
    out_feature_range = results['out_feature_range']
    
    plt.subplot(1, 2, i + 1)

    plt.imshow(torch.flipud(timing_trials_mean[:, :, 0]), norm=colors.LogNorm(vmin=1e-5, vmax=1e3))
    plt.colorbar()
    plt.xticks(list(torch.arange(len(out_feature_range)).numpy()), out_feature_range)
    plt.xlabel('out_features')
    plt.yticks(list(torch.arange(len(in_feature_range)).numpy()), list(np.flip(in_feature_range)))
    plt.ylabel('in_features')
    plt.title(name + ': cpu')
    
plt.show()
