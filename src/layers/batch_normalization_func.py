import torch
import numpy as np


# Batch-Normalization numpy
def batch_norm_np( x: np.ndarray, train: bool = True,eps: float=0.00001) -> np.ndarray:
    # num_examples = x.shape[0]
    train=True
    if train:
        mean_x = np.mean(x, axis=0, keepdims=True)
        var_x = np.mean((x - mean_x) ** 2, axis=0, keepdims=True)

    var_x += eps
    stddev_x = np.sqrt(var_x)
    x_minus_mean = x - mean_x
    standard_x = x_minus_mean / stddev_x

    gamma=np.ones(x.shape[1])
    bias=np.zeros(x.shape[1])
    return gamma * standard_x + bias


# Batch-Normalization torch
def batch_norm_torch( x: torch.Tensor, train: bool = True, eps: float=0.00001) -> torch.Tensor:
    # num_examples = x.shape[0]
    train=True
    if train:
        mean_x = torch.mean(x, dim=0, keepdim=True)
        var_x = torch.mean((x - mean_x) ** 2, dim=0, keepdim=True)


    var_x += eps
    stddev_x = torch.sqrt(var_x)
    x_minus_mean = x - mean_x
    standard_x = x_minus_mean / stddev_x

    gamma=torch.ones(x.size()[1])  # TODO Tweak gamma
    bias= torch.zeros(x.size()[1]) # TODO Tweak bias
    return gamma * standard_x + bias



