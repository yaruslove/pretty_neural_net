import torch
import torch.nn as nn
import numpy as np


import sys
# appending a path
sys.path.append('../../src/layers')
from BatchNorm1d import CustomBatchNorm1d


## TORCH LIBRARY
input_size = 3
batch_size = 5
eps = 1e-1

torch_batch_norm1d = nn.BatchNorm1d(input_size, eps=eps)
torch_batch_norm1d.bias.data = torch.randn(input_size, dtype=torch.float)
torch_batch_norm1d.weight.data = torch.randn(input_size, dtype=torch.float)
torch_batch_norm1d.momentum = 0.5



## INPUT
torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
torch_input


# TRAIN MODE 

## My BatchNorm1d
my_batch_norm1d = CustomBatchNorm1d(torch_batch_norm1d.weight.data,
                                        torch_batch_norm1d.bias.data, eps, torch_batch_norm1d.momentum)
my_batch_norm1d_output_train=my_batch_norm1d(torch_input)
print(f"my_batch_norm1d_output_train: {my_batch_norm1d_output_train}")

## TORCH LIB BatchNorm1d
torch_batch_norm1d_output_train=torch_batch_norm1d(torch_input)
print(f"torch_batch_norm1d_output_train: {torch_batch_norm1d_output_train}")

# Compare answers
isEqual_train_outputs=torch.allclose(my_batch_norm1d_output_train, torch_batch_norm1d_output_train)
print(f"TRAIN MODE. My layer and torch norm1d are equal output: {isEqual_train_outputs}")




# EVAL MODE
my_batch_norm1d.eval()
torch_batch_norm1d.eval()

## Inference
my_batch_norm1d_output_eval=my_batch_norm1d(torch_input)
torch_batch_norm1d_output_eval=torch_batch_norm1d(torch_input)

print(f"my_batch_norm1d_output_eval: {my_batch_norm1d_output_eval}")
print(f"torch_batch_norm1d_output_eval: {torch_batch_norm1d_output_eval}")


# Compare answers
isEqual_eval_outputs=torch.allclose(my_batch_norm1d_output_eval, torch_batch_norm1d_output_eval)
print(f"EVAL MODE. My layer and torch norm1d are equal output: {isEqual_eval_outputs}")
