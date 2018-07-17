import os
import torch
import torch.nn as nn
import torch.nn.functional as F

noiseFeature=30
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.fc1=nn.Linear(noiseFeature,1024)
        self.fc2=nn.Linear(1024,28*28)


    def forward(self,x):
        x=F.leaky_relu(self.fc1(x),0.2)
        x=F.sigmoid(self.fc2(x))
        return x


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.fc1=nn.Linear(28*28,1024)
        self.fc2=nn.Linear(1024,1)

    def forward(self,x):
        x=F.leaky_relu(self.fc1(x),0.2)
        x=F.sigmoid(self.fc2(x))
        return x
