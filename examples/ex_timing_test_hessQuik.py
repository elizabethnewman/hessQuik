import torch
from examples.timing_test import timing_test

in_feature_range = (2 ** torch.arange(1, 6)).tolist()
out_feature_range = (2 ** torch.arange(0, 1)).tolist()
nex_range = [10]
width = 20
depth = 4
num_trials = 10
seed = 1234


torch.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
network_type = 'PytorchHessian'
reverse_mode = False

results = timing_test(in_feature_range, out_feature_range, nex_range,
                      num_trials=num_trials, width=width, depth=depth,
                      network_type=network_type, device=device, clear_memory=True,
                      reverse_mode=reverse_mode)

print(results['timing_trials_mean'].squeeze())
