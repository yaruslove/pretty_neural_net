import torch
from src.layers.BatchNorm1d import CustomBatchNorm1d

weight_data=torch.tensor([ 0.2961,  1.2503, -0.1758])
bias_data=torch.tensor([0.6596, 1.6274, 0.9150])
input_size = 3
batch_size = 5
eps = 1e-1
momentum = 0.5

my_batch_norm1d = CustomBatchNorm1d(weight_data,
                                    bias_data, 
                                    eps, 
                                    momentum)

torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
torch_input
# TRAIN MODE
torch_output=my_batch_norm1d(torch_input)
# EVAL MODE
my_batch_norm1d.eval()
torch_output=my_batch_norm1d(torch_input)