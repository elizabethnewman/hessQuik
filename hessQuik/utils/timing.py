import torch
import hessQuik
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
import time
import gc
from typing import Union


def setup_device_and_gradient(f: hessQuik.networks.NN, network_wrapper: str = 'hessQuik', device: str = 'cpu') \
        -> torch.nn.Module:
    r"""
    Setup network with correct wrapper and device
    """
    # map to device
    f = f.to(device)

    if network_wrapper == 'PytorchAD':
        f = net.NNPytorchAD(f)

    if network_wrapper == 'PytorchHessian':
        f = net.NNPytorchHessian(f)

    return f


def setup_resnet(in_features: int, out_features: int, width: int = 16, depth: int = 4) -> hessQuik.networks.NN:
    r"""
    Setup resnet architecture for timing tests
    """

    f = net.NN(lay.singleLayer(in_features, width, act=act.antiTanhActivation()),
               net.resnetNN(width, depth, h=0.5, act=act.tanhActivation()),
               lay.singleLayer(width, out_features, act=act.identityActivation()))
    return f


def setup_fully_connected(in_features: int, out_features: int, width: int = 16, depth: int = 4) -> hessQuik.networks.NN:
    r"""
    Setup fully-connected architecture for timing tests
    """
    f = net.fullyConnectedNN([in_features] + depth * [width] + [out_features], act=act.tanhActivation())
    return f


def setup_icnn(in_features: int, out_features: int, width: int = 16, depth: int = 4) -> hessQuik.networks.NN:
    r"""
    Setup ICNN architecture for timing tests.

    Requires scalar output.
    """

    if out_features > 1:
        raise ValueError('Must be scalar output for ICNN example')

    f = net.NN(net.ICNN(in_features, [None] + depth * [width], act=act.tanhActivation()),
               lay.quadraticICNNLayer(in_features, width, rank=2))
    return f


def setup_network(in_features: int, out_features: int, width: int, depth: int, network_type: str = 'resnet',
                  network_wrapper: str = 'hessQuik', device: str = 'cpu'):
    r"""
    Wrapper to setup network.
    """
    if network_type == 'resnet':
        f = setup_resnet(in_features, out_features, width, depth)
    elif network_type == 'fully_connected':
        f = setup_fully_connected(in_features, out_features, width, depth)
    elif network_type == 'icnn':
        f = setup_icnn(in_features, out_features, width, depth)
    else:
        raise ValueError('network_type must be "resnet", "fully_connected", or "icnn"')

    f = setup_device_and_gradient(f, network_wrapper=network_wrapper, device=device)

    return f


def timing_test_cpu(f: Union[hessQuik.networks.NN, torch.nn.Module], x: torch.Tensor, num_trials: int = 10,
                    clear_memory: bool = True) -> torch.Tensor:
    r"""
    Timing test for one architecture on CPU.

    Test is run ``num_trials`` times and the timing for each trial is returned.

    The timing includes one dry run for the first iteration that is not returned.

    Memory is cleared after the completion of all trials.
    """
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


def timing_test_gpu(f: Union[hessQuik.networks.NN, torch.nn.Module], x: torch.Tensor,
                    num_trials: int = 10, clear_memory: bool = True):
    r"""
    Timing test for one architecture on CPU.

    Test is run ``num_trials`` times and the timing for each trial is returned.

    Each trial includes a ``torch.cuda.synchonize`` call.

    The timing includes one dry run for the first iteration that is not returned.

    Memory is cleared after the completion of all trials.
    """

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


def timing_test(in_feature_range: torch.Tensor, out_feature_range: torch.Tensor, nex: int = 10, num_trials: int = 10, 
                width: int = 16, depth: int = 4, network_wrapper: str = 'hessQuik', 
                network_type: str = 'resnet', device: str = 'cpu', clear_memory: bool = True) -> dict:
    r"""
    
    :param in_feature_range: available input feature dimensions
    :type in_feature_range: torch.Tensor
    :param out_feature_range: available output feature dimensions
    :type out_feature_range: torch.Tensor
    :param nex: number of examples for network input. Default: 10
    :type nex: int, optional
    :param num_trials: number of trials per input-output feature combination. Default: 10
    :type num_trials: int, optional
    :param width: width of network. Default: 16
    :type width: int, optional
    :param depth: depth of network. Default: 4
    :type depth: int, optional
    :param network_wrapper: type of network wrapper. Default: 'hessQuik'. Options: 'hessQuik', 'PytorchAD', 'PytorchHessian'
    :type network_wrapper: str, optional
    :param network_type: network architecture. Default: 'resnet'. Options: 'resnet', 'fully_connected', 'icnn'
    :type network_type: str, optional
    :param device: device for testing. Default: 'cpu'
    :type device: str, optional
    :param clear_memory: flag to clear memory after each set of trials
    :type clear_memory: bool, optional
    :return: dictionary containing keys

        - **'timing_trials'** (*torch.Tensor*) - time (in seconds) for each trial and each architecture
        - **'timing_trials_mean'** (*torch.Tensor*) - average over trials for each architecture
        - **'timing_trials_std'** (*torch.Tensor*) - standard deviation over trials for each architecture
        - **'in_feature_range'** (*torch.Tensor*) - available input feature dimensions
        - **'out_feature_range'** (*torch.Tensor*) - available output feature dimensions
        - **'nex'** (*int*) - number of samples for the input data
        - **'num_trials'** (*int*) - number of trials per architecture

    """

    # initialize
    timing_trials = torch.zeros(len(in_feature_range), len(out_feature_range), num_trials)
    timing_trials_mean = torch.zeros(len(in_feature_range), len(out_feature_range))
    timing_trials_std = torch.zeros_like(timing_trials_mean)

    for idx_out, out_features in enumerate(out_feature_range):
        for idx_in, in_features in enumerate(in_feature_range):

            # setup network
            f = setup_network(in_features, out_features, width, depth,  network_type=network_type,
                              network_wrapper=network_wrapper, device=device)

            # setup data
            x = torch.randn(nex, in_features, device=device)

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
