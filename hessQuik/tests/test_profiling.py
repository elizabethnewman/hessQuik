import torch
import cProfile
import profile
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net


nex = 100
d = 100
width = 500
x = torch.randn(nex, d)
f = lay.singleLayer(d, 32, act=act.identityActivation())

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

print('------ torch BACKWARD ------')
with torch.profiler.profile() as prof:
    with torch.profiler.record_function("singleLayer"):
        f(x, do_gradient=True, do_Hessian=True, reverse_mode=True)
        f.backward(do_Hessian=True)

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
