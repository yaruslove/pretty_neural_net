import torch
import torch.nn as nn
import numpy as np


import sys
# appending a path
sys.path.append('../../src/layers')
from Conv2dLoop import Conv2dLoop

# Create batch from images with some chanels
batch_imgs = torch.randint(10, (2, 3, 7, 10)).to(torch.float32)  # .to(torch.float32)
print(f"batch_imgs.shape {batch_imgs.shape}")


# Create tensor
my_kernel = torch.randint(10, (4, 3, 3, 3)).to(torch.float32)
print(f"my_kernel.shape {my_kernel.shape}")


# Conv params
in_channels = 3
out_channels = 4
kernel_size = 3
stride = 1

# Cheking kernel size should be same as kernel matrix sample
kernel_hight = my_kernel.shape[-1]
kernel_widith = my_kernel.shape[-2]
assert kernel_hight == kernel_widith == kernel_size, 'Error Kernel convolution is not square size!!!'


###### MY LAYER CONV2d ######
my_conv2d = Conv2dLoop(in_channels, out_channels, kernel_size, stride)
# SETTING KERNEL WIEGHTS
my_conv2d.set_kernel(my_kernel)
# INFERENCE
my_conv2d_out = my_conv2d(batch_imgs)
print(f"my_conv2d_out: {my_conv2d_out}")




###### TORCH LIBRARY CONV2d ######
conv2d_torch = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding=0, bias=False)
# SETTING KERNEL WIEGHTS
conv2d_torch.weight.data = my_kernel

torch_conv2d_out=conv2d_torch(batch_imgs)
print(f"torch_conv2d_out: {torch_conv2d_out}")


print (f"Is equal my_conv2d and torch_conv2d: {torch.allclose(my_conv2d_out, torch_conv2d_out, atol=1e-08)} ")