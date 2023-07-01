import torch
import torch.nn as nn
import numpy as np


import sys
# appending a path
sys.path.append('../../src/layers')
from batch_normalization import batch_norm_np, batch_norm_torch

# Gen data
x_torch=torch.randint(10, (3, 7)).to(torch.float32)
x_np=x_torch.numpy()
eps = np.power(10., -3)
print(f"input data {x_torch}")




# Numpy exampel
out_np=batch_norm_np(x_np,True,eps)
print(f"out_np {out_np}")

# Torch exampel
out_torch=batch_norm_torch(x_torch,True,eps)
print(f"out_torch {out_torch}")

# Torch lib
batch_norm_lib = nn.BatchNorm1d(x_torch.shape[1], affine=False)
batch_norm_lib.eps = eps
out_torchlib=batch_norm_lib(x_torch)
print(f"out_torchlib {out_torchlib}")

# convert numpy to torch
out_np_torch= torch.from_numpy(out_np).to(torch.float32)
answer=torch.allclose(out_np_torch, out_torch) and torch.allclose(out_np_torch, out_torchlib) and torch.allclose(out_torch, out_torchlib)

print(f"cheking Tensors are equal: {answer}")