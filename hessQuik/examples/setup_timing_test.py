import torch
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
from time import time
import gc


def create_network(in_features, out_features, width=20, depth=4,
                   network_type='hessQuik', device='cpu'):

    f = net.NN(lay.singleLayer(in_features, width, act=act.antiTanhActivation()),
               net.resnetNN(width, depth, h=0.5, act=act.tanhActivation()),
               lay.singleLayer(width, out_features, act=act.identityActivation())).to(device)

    # f = net.fullyConnectedNN([in_features] + depth * [width] + [out_features], act=act.tanhActivation())

    if network_type == 'hessQuik':
        x_requires_grad = False

    elif network_type == 'PytorchAD':
        f = net.NNPytorchAD(f)
        x_requires_grad = True

    elif network_type == 'PytorchHessian':
        f = net.NNPytorchHessian(f)
        x_requires_grad = True

    else:
        raise ValueError('network_type must be "hessQuik", "PytorchAD", or "PytorchHessian"')

    return f, x_requires_grad


def timing_test_forward(f, x, num_trials=10, clear_memory=True):

    total_time = torch.zeros(num_trials)
    for i in range(num_trials):
        t1_start = time()
        f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=True)
        t1_stop = time()
        total_time[i] = t1_stop - t1_start

    if clear_memory:
        torch.cuda.empty_cache()
        del f, x
        gc.collect()

    return total_time


def timing_test(in_feature_range, out_feature_range, nex_range, num_trials=10, width=20, depth=4,
                             network_type='hessQuik', device='cpu', clear_memory=True):

    # initialize
    timing_trials = torch.zeros(len(in_feature_range), len(out_feature_range), len(nex_range), num_trials)
    timing_trials_mean = torch.zeros(len(in_feature_range), len(out_feature_range), len(nex_range))
    timing_trials_std = torch.zeros_like(timing_trials_mean)

    for idx_out, out_features in enumerate(out_feature_range):
        for idx_in, in_features in enumerate(in_feature_range):
            for idx_nex, nex in enumerate(nex_range):

                # setup network
                f, x_requires_grad = create_network(in_features, out_features, width, depth,
                                                    network_type=network_type, device=device)
                x = torch.randn(nex, in_features, device=device)
                x.requires_grad = x_requires_grad

                # main test
                total_time = timing_test_forward(f, x, num_trials=num_trials, clear_memory=clear_memory)

                timing_trials[idx_in, idx_out, idx_nex] = total_time
                timing_trials_mean[idx_in, idx_out, idx_nex] = torch.mean(total_time).item()
                timing_trials_std[idx_in, idx_out, idx_nex] = torch.std(total_time).item()

                del f, x
                gc.collect()

    results = {'timing_trials': timing_trials,
               'timing_trials_mean': timing_trials_mean,
               'timing_trials_std': timing_trials_std,
               'in_feature_range': in_feature_range,
               'out_feature_range': out_feature_range,
               'nex_range': nex_range,
               'num_trials': num_trials}
    return results
