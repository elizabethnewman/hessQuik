import torch
import pickle
from examples.timing_test import timing_test
import argparse
from datetime import datetime


parser = argparse.ArgumentParser(description='hessQuik-timing')

parser.add_argument('--num-input', type=int, default=3, metavar='n',
                    help='number of different input features, given by powers of 2 starting from 2^0 '
                         '(default: 3 giving input features (1, 2, 4)')
parser.add_argument('--num-output', type=int, default=1, metavar='m',
                    help='number of different output features, given by powers of 2 starting from 2^0 '
                         '(default: 1 giving output features (1,)')
parser.add_argument('--num-examples', type=int, default=1, metavar='e',
                    help='number of examples, given by multiples of of 10, starting with 10'
                         '(default: 1 giving number of examples (10,)')
parser.add_argument('--num-trials', type=int, default=10, metavar='N', help='number of trials (default: 10)')
parser.add_argument('--seed', type=int, default=42, metavar='s', help='random seed (default: 42)')
parser.add_argument('--width', type=int, default=20, metavar='w', help='width of network (default: 20)')
parser.add_argument('--depth', type=int, default=4, metavar='d', help='depth of network (default: 4)')
parser.add_argument('--network-type', type=str, default='hessQuik', metavar='t',
                    help='type of network (default: "hessQuiK"), '
                         'options include ("hessQuiK", "PytorchAD", "PytorchHessian")')
parser.add_argument('--reverse-mode', action='store_true', default=False,
                    help='type of network (default: "hessQuik"), '
                         'options include ("hessQuik", "PytorchAD", "PytorchHessian")')
parser.add_argument('--save', action='store_true', default=False, help='save results')

args = parser.parse_args()

# setup
in_feature_range = (2 ** torch.arange(0, args.num_input)).tolist()
out_feature_range = (2 ** torch.arange(0, args.num_output)).tolist()
nex_range = (10 * torch.arange(1, args.num_examples + 1)).tolist()
width = args.width
depth = args.depth
num_trials = args.num_trials
seed = args.seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'
network_type = args.network_type
reverse_mode = args.reverse_mode


# filename
now = datetime.now()
my_date = now.strftime("%m-%d-%Y--")

mode = 'forward'
if reverse_mode:
    mode = 'backward'
filename = network_type + '-' + mode + '-' + device
print(my_date + filename)


# main
torch.manual_seed(seed)
results = timing_test(in_feature_range, out_feature_range, nex_range,
                      num_trials=num_trials, width=width, depth=depth,
                      network_type=network_type, device=device, clear_memory=True,
                      reverse_mode=reverse_mode)

if args.save:
    pickle.dump(results, open(my_date + '--' + filename + ".p", "wb"))
