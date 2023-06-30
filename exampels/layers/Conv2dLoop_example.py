import torch
import torch.nn as nn
import numpy as np


import sys
# appending a path
sys.path.append('../../src/layers')
from Conv2dLoop import Conv2dLoop

# Create batch from images with some chanels
img = torch.randint(10, (2, 3, 7, 10)).to(torch.float32)  # .to(torch.float32)
print(img.shape)


# Create tensor
my_kernel = torch.randint(10, (4, 3, 3, 3)).to(torch.float32)
print(my_kernel.shape)


in_channels = 3
out_channels = 4
kernel_size = 3
stride = 1

# Cheking kernel size should be same as kernel matrix sample
kernel_hight = my_kernel.shape[-1]
kernel_widith = my_kernel.shape[-2]
assert kernel_hight == kernel_widith == kernel_size, 'Error Kernel convolution isnot  square size!!!'

my_conv = Conv2dLoop(in_channels, out_channels, kernel_size, stride)
# my_conv.set_kernel(kernel)
custom_conv2d_out=my_conv.set_kernel(my_kernel)
print(f"custom_conv2d_out {custom_conv2d_out}")



###### TORCH LIBRARY CONV2d ######
conv2d_torch = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding=0, bias=False)
# SETTING KERNEL WIEGHTS
conv2d_torch.weight.data = my_kernel

conv2d_out=conv2d_torch(img)
print(f"conv2d_out {conv2d_out}")


torch.allclose(custom_conv2d_out, conv2d_out)