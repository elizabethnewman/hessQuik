import torch
import cProfile
import profile
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net
from line_profiler import LineProfiler

nex = 100
d = 1
width = 100

# f = net.NN(lay.singleLayer(d, width, act=act.tanhActivation()),
#            lay.singleLayer(width, 32, act=act.tanhActivation()))
f = net.fullyConnectedNN([d, width, 1], act=act.antiTanhActivation())
x = torch.randn(nex, f.dim_input())


def eval():
    f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=True, forward_mode=True)


lp = LineProfiler()
lp.timer_unit = 1e-6
lp.add_function(eval)
# lp.add_function(f[0].forward)
lp.print_stats()

# nex = 100
# in_features = 4
# width = 100
# depth = 4
# out_features = 1
# f = net.NN(lay.singleLayer(in_features, width, act=act.identityActivation()),
#            net.resnetNN(width, depth, h=0.5, act=act.identityActivation()),
#            lay.singleLayer(width, out_features, act=act.identityActivation()))
#
# x = torch.randn(nex, f.dim_input())

# f = net.NNPytorchAD(f)
# x.requires_grad = True

# print('------ cProfile FORWARD ------')
# cProfile.run('f(x, do_gradient=True, do_Hessian=True)', sort='tottime')

# print('------ profile FORWARD------')
# profile.run('f(x, do_gradient=True, do_Hessian=True)', sort='tottime')

# print('------ torch FORWARAD ------')
# with torch.profiler.profile() as prof:
#     with torch.profiler.record_function("singleLayer"):
#         f(x, do_gradient=True, do_Hessian=True)
#
# print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

# print('------ torch  ------')
# with torch.profiler.profile() as prof:
#     with torch.profiler.record_function("singleLayer"):
#         f0, df0, d2f0 = f(x, do_gradient=True, do_Hessian=True)
#
# print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))


# if __name__ == '__main__':
#     eval()
