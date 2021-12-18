import torch
import pickle
from hessQuik.utils import timing_test
import argparse
from datetime import datetime
import os

# -------------------------------------------------------------------------------------------------------------------- #
# create parser
parser = argparse.ArgumentParser(description='hessQuik-timing')
parser.add_argument('--num-input',          type=int,               default=1,              metavar='n',
                    help='number of input features by powers of 2; start from 2^0 (default: 1 input features (2^0,)')
parser.add_argument('--num-output',         type=int,               default=1,              metavar='m',
                    help='number of input features by powers of 2; start from 2^0 (default: 1 input features (2^0,)')
parser.add_argument('--num-examples',       type=int,               default=4,              metavar='e',
                    help='number of threads (default: 4)')
parser.add_argument('--num-trials',         type=int,               default=10,             metavar='N',
                    help='number of trials (default: 10)')
parser.add_argument('--num-threads',        type=int,               default=10,             metavar='N',
                    help='number of trials (default: 10)')
parser.add_argument('--seed',               type=int,               default=42,             metavar='s',
                    help='random seed (default: 42)')
parser.add_argument('--width',              type=int,               default=16,             metavar='w',
                    help='width of network (default: 16)')
parser.add_argument('--depth',              type=int,               default=4,              metavar='d',
                    help='depth of network (default: 4)')
parser.add_argument('--network-type',       type=str,               default='resnet',     metavar='t',
                    help='type of network from ("resnet", "fully_connected", "icnn") (default: "resnet")')
parser.add_argument('--network-wrapper',    type=str,               default='hessQuik',     metavar='t',
                    help='network wrapper from ("hessQuiK", "PytorchAD", "PytorchHessian") (default: "hessQuiK")')
parser.add_argument('--save',               action='store_true',    default=False,
                    help='save results')
parser.add_argument('--verbose',            action='store_true',    default=False,
                    help='print mean results')
args = parser.parse_args()

# -------------------------------------------------------------------------------------------------------------------- #
# setup parameters
in_feature_range = (2 ** torch.arange(0, args.num_input)).tolist()
out_feature_range = (2 ** torch.arange(0, args.num_output)).tolist()
nex = args.num_examples
width = args.width
depth = args.depth
num_trials = args.num_trials
seed = args.seed
network_type = args.network_type
network_wrapper = args.network_wrapper

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(args.num_threads)
print(args)
print('number of computational threads: %d' % torch.get_num_threads())
# -------------------------------------------------------------------------------------------------------------------- #
# filename
now = datetime.now()
my_date = now.strftime("%m-%d-%Y-")

filename = (network_wrapper + '-' + network_type + '-' + device
            + '-w' + str(width) + '-d' + str(depth) + '-out' + str(len(out_feature_range)))
print(my_date + filename)

# -------------------------------------------------------------------------------------------------------------------- #
# main
torch.manual_seed(seed)

# warm up
warm_up = timing_test(in_feature_range, out_feature_range, nex,
                      num_trials=2, width=width, depth=depth,
                      network_wrapper=network_wrapper, network_type=network_type, device=device, clear_memory=True)

# main run
results = timing_test(in_feature_range, out_feature_range, nex,
                      num_trials=num_trials, width=width, depth=depth,
                      network_wrapper=network_wrapper,  network_type=network_type, device=device, clear_memory=True)

if args.verbose:
    print('in_feature_range: ', results['in_feature_range'])
    print('out_feature_range: ', results['out_feature_range'])
    print('size of mean timing: ', results['timing_trials_mean'].shape)
    print('mean timing (in features x out_features):\n', results['timing_trials_mean'])

# -------------------------------------------------------------------------------------------------------------------- #
if args.save:

    dir_name = 'results/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    results = {'results': results, 'args': args}
    pickle.dump(results, open(dir_name + my_date + '-' + filename + ".p", "wb"))

