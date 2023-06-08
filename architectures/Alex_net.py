import torch
import torch.nn as nn

class LinNormAct(torch.nn.Module):
    """
    #args[0] -- число входных каналов
    #args[1] -- число выходных каналов
    """
    def __init__(self,
                 *args,
                 activation=torch.nn.ReLU,
                 normalization=torch.nn.BatchNorm1d,
                 **kwargs,):
        super().__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(*args, **kwargs),
            normalization(args[1]),
            activation()
        )

    def forward(self, input):
        return self.module(input)       
        
class ConvNormAct(torch.nn.Module):
    """
    #args[0] -- число входных каналов
    #args[1] -- число выходных каналов
    #args[2] -- размер ядра (kernel size)
    #args[3] -- stride 
    """
    def __init__(self,
                 *args,
                 activation=torch.nn.ReLU,
                 normalization=torch.nn.BatchNorm2d,
                 **kwargs):
        super().__init__()
        if not "stride" in kwargs:
            kwargs["padding"] = (args[2] - 1) // 2

        self.module = torch.nn.Sequential(
            torch.nn.Conv2d(*args, **kwargs),
            normalization(args[1]),
            activation()
        )

    def forward(self, input):
        return self.module(input)

class AlexNet(torch.nn.Module):
    def __init__(self,
                 num_classes=100,
                 normalization2d=torch.nn.BatchNorm2d,
                 normalization1d=torch.nn.BatchNorm1d,
                 activation=torch.nn.ReLU):
        super().__init__()

        actnorm2d = {'activation': activation, 'normalization': normalization2d}
        actnorm1d = {'activation': activation, 'normalization': normalization1d}
        
        self.conv = torch.nn.Sequential(
            ConvNormAct(3, 32, 5, stride=2, padding=2, **actnorm2d), # ConvNormAct(3, 32, 5, stride=2, padding=2, **actnorm2d)
            torch.nn.MaxPool2d(2),
            ConvNormAct(32, 64, 3, **actnorm2d),
            torch.nn.MaxPool2d(2),
            ConvNormAct(64, 128, 3, **actnorm2d),
            ConvNormAct(128, 128, 3, **actnorm2d),
            ConvNormAct(128, 128, 3, **actnorm2d),
            torch.nn.MaxPool2d(2)
        )
        
        self.fc = torch.nn.Sequential(
            LinNormAct(128*4*4, 1024, **actnorm1d),
            LinNormAct(1024, 512, **actnorm1d),
            torch.nn.Linear(512, num_classes)
        )
        
        
    def forward(self, input):
        res = self.conv(input)
        res = torch.flatten(res, 1)
        res = self.fc(res)
        return res
    

x=torch.rand(7, 3, 64, 64)
model = AlexNet()
out=model(x)
print(out.size())
        