import torch
import torch.nn as nn


input_size = 3
batch_size = 5
eps = 1e-1


class CustomBatchNorm1d:
    def __init__(self, weight, bias, eps, momentum):
        # Реализуйте в этом месте конструктор.
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.momentum = momentum
        self.ema_mean_x = torch.zeros(1)
        self.ema_std_x = torch.ones(1)
        self.__valid=False
        
    def __call__(self, x):
        # normed_tensor = # Напишите в этом месте нормирование входного тензора.
        if self.__valid==True:
            mean_x = self.ema_mean_x.clone()
            std_x = self.ema_std_x.clone()
        elif self.__valid==False:
            mean_x = torch.mean(x, dim=0, keepdim=True)
            std_x = torch.mean((x - mean_x) ** 2, dim=0, keepdim=True)
            n=x.size()[0]
            self.update_running_variables(mean_x,std_x,n)

        stddev_x = torch.sqrt(std_x+self.eps)
        standard_x = (x - mean_x) / stddev_x

        return self.weight * standard_x + self.bias
        
    def eval(self):
        self.__valid=True


    def update_running_variables(self, mean_x, std_x, n):

        # n=std_x.size()[1]
        std_x_bias=((std_x*n)/(n-1)).clone() # НЕсмещеная дисперсия

        self.ema_mean_x = self.momentum * mean_x + \
                          (1.0 - self.momentum) * self.ema_mean_x
        self.ema_std_x = self.momentum * std_x_bias + \
                         (1.0 - self.momentum) * self.ema_std_x