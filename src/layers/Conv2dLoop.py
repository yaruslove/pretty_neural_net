import torch
from abc import ABC, abstractmethod


class ABCConv2d(ABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def set_kernel(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, input_tensor):
        pass

# Сверточный слой через циклы.
## Ограничение работы моего сверточного слоя: 
'''
1. padding = 0
2. Размерность ядра свертки - kernel должен быть квадрат 
    (высота ядра = ширине ядра) 
    kernel.shape[-1] = kernel.shape[-2] = kernel_height =kernel_height 
3. Нет алгортима backwards
'''



class Conv2dLoop(ABCConv2d):

    def create_empty_out_img(self, input_tensor):
        input_width = input_tensor.shape[-1]
        input_height = input_tensor.shape[-2]
        padding = 0
        
        self.output_height = (input_height + 2 * padding - (self.kernel_size - 1) - 1) // self.stride + 1
        self.output_width = (input_width + 2 * padding - (self.kernel_size - 1) - 1) // self.stride + 1
        
        # Cheking batch size 
        if len(input_tensor.shape) == 3: 
            batch_size = 1
        if len(input_tensor.shape) == 4:
            batch_size = input_tensor.shape[-4]

        out_tesor = torch.zeros([ batch_size, self.out_channels, self.output_height, self.output_width])
        return out_tesor

    def conv_one_sample(self, one_sample, filter, idx_batch, idx_filter):
        h_start = 0
        for _ in range(self.output_height): #h_centr
            w_start = 0
            for _ in range(self.output_width): # w_centr
                # window_sliced - окно которое бежит по вертикали и горизонтали
                window_sliced = one_sample[:,   h_start:h_start+self.kernel_size,   w_start: w_start+self.kernel_size]
                self.out_tensor[ idx_batch, idx_filter, h_start, w_start] = torch.sum( window_sliced * self.kernel[idx_filter])
                w_start+=self.stride
            h_start+=self.stride


    def __call__(self, input_tensor):
        # Вычисление свертки с использованием циклов.
        self.out_tensor=self.create_empty_out_img(input_tensor)

        # Check tensor is batch or not and unsqueeze if not
        if len(input_tensor.shape)==4:
            is_batch = True
        elif len(input_tensor.shape)==3:
            is_batch = False
            input_tensor = input_tensor.unsqueeze(0)

        # main loops
        for idx_batch, one_sample in enumerate(input_tensor):   # itteration through batch 
            for idx_filter, filter in enumerate(self.kernel):   # itteration through kernel filters 
                self.conv_one_sample(one_sample, filter, idx_batch, idx_filter)

        # squeeze if not batch
        if is_batch:
            self.out_tensor = torch.squeeze(self.out_tensor,dim=0)
        
        return self.out_tensor