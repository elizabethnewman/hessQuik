import torch
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
import time
import gc


def setup_device_and_gradient(f, network_wrapper='hessQuik', device='cpu'):

    # map to device
    f = f.to(device)

    # determine if x requires gradient
    if network_wrapper == 'hessQuik':
        x_requires_grad = False

    elif network_wrapper == 'PytorchAD':
        f = net.NNPytorchAD(f)
        x_requires_grad = True

    elif network_wrapper == 'PytorchHessian':
        f = net.NNPytorchHessian(f)
        x_requires_grad = True

    else:
        raise ValueError('network_wrapper must be "hessQuik", "PytorchAD", or "PytorchHessian"')

    return f, x_requires_grad


def setup_resnet(in_features, out_features, width=20, depth=4):

    f = net.NN(lay.singleLayer(in_features, width, act=act.antiTanhActivation()),
               net.resnetNN(width, depth, h=0.5, act=act.tanhActivation()),
               lay.singleLayer(width, out_features, act=act.identityActivation()))
    return f


def setup_fully_connected(in_features, out_features, width=20, depth=4):
    f = net.fullyConnectedNN([in_features] + depth * [width] + [out_features], act=act.tanhActivation())
    return f


def setup_icnn(in_features, out_features, width=20, depth=4):
    if out_features > 1:
        raise ValueError('Must be scalar output for ICNN example')

    f = net.NN(net.ICNN(in_features, [None] + depth * [width], act=act.tanhActivation()),
               lay.quadraticICNNLayer(in_features, width, rank=2))
    return f


def setup_network(in_features, out_features, width, depth, network_type='resnet',
                  network_wrapper='hessQuik', device='cpu'):
    if network_type == 'resnet':
        f = setup_resnet(in_features, out_features, width, depth)
    elif network_type == 'fully_connected':
        f = setup_fully_connected(in_features, out_features, width, depth)
    elif network_type == 'icnn':
        f = setup_icnn(in_features, out_features, width, depth)
    else:
        raise ValueError('network_type must be "resnet", "fully_connected", or "icnn"')

    f, x_requires_grad = setup_device_and_gradient(f, network_wrapper=network_wrapper, device=device)

    return f, x_requires_grad


def timing_test_cpu(f, x, num_trials=10, clear_memory=True):

    total_time = torch.zeros(num_trials + 1)
    for i in range(num_trials + 1):
        t1_start = time.perf_counter()
        f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=True)
        t1_stop = time.perf_counter()
        total_time[i] = t1_stop - t1_start

    if clear_memory:
        del f, x
        gc.collect()
        torch.cuda.empty_cache()

    return total_time[1:]


def timing_test_gpu(f, x, num_trials=10, clear_memory=True):

    total_time = torch.zeros(num_trials + 1)
    for i in range(num_trials + 1):
        t1_start = time.perf_counter()
        f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=True)
        torch.cuda.synchronize()
        t1_stop = time.perf_counter()
        total_time[i] = t1_stop - t1_start

    if clear_memory:
        del f, x
        gc.collect()
        torch.cuda.empty_cache()

    return total_time[1:]


def timing_test(in_feature_range, out_feature_range, nex=10, num_trials=10, width=20, depth=4,
                network_wrapper='hessQuik', network_type='resnet', device='cpu', clear_memory=True):

    # initialize
    timing_trials = torch.zeros(len(in_feature_range), len(out_feature_range), num_trials)
    timing_trials_mean = torch.zeros(len(in_feature_range), len(out_feature_range))
    timing_trials_std = torch.zeros_like(timing_trials_mean)

    for idx_out, out_features in enumerate(out_feature_range):
        for idx_in, in_features in enumerate(in_feature_range):

            # setup network
            f, x_requires_grad = setup_network(in_features, out_features, width, depth,  network_type=network_type,
                                               network_wrapper=network_wrapper, device=device)

            # setup data
            x = torch.randn(nex, in_features, device=device)
            x.requires_grad = x_requires_grad

            # main test
            if device == 'cpu':
                total_time = timing_test_cpu(f, x, num_trials=num_trials, clear_memory=clear_memory)
            else:
                total_time = timing_test_gpu(f, x, num_trials=num_trials, clear_memory=clear_memory)

            # store results
            timing_trials[idx_in, idx_out] = total_time
            timing_trials_mean[idx_in, idx_out] = torch.mean(total_time).item()
            timing_trials_std[idx_in, idx_out] = torch.std(total_time).item()

    # store setup and results
    results = {'timing_trials': timing_trials,
               'timing_trials_mean': timing_trials_mean,
               'timing_trials_std': timing_trials_std,
               'in_feature_range': in_feature_range,
               'out_feature_range': out_feature_range,
               'nex': nex,
               'num_trials': num_trials}

    return results
