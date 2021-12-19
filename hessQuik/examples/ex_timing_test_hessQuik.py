import torch
from hessQuik.utils import timing_test

in_feature_range = (2 ** torch.arange(0, 5)).tolist()
out_feature_range = (2 ** torch.arange(0, 1)).tolist()
nex = 10
width = 16
depth = 4
num_trials = 10
seed = 1234


torch.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
network_wrapper = 'PytorchHessian'
network_type = 'resnet'
print(network_type)

# warm up
results = timing_test(in_feature_range, out_feature_range, nex,
                      num_trials=2, width=width, depth=depth,
                      network_type=network_type, device=device, clear_memory=True)
print(results['timing_trials_mean'].squeeze())

results = timing_test(in_feature_range, out_feature_range, nex,
                      num_trials=num_trials, width=width, depth=depth,
                      network_type=network_type, device=device, clear_memory=True)

print(results['timing_trials_mean'].squeeze())
