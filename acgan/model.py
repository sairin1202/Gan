import os
import torch
import torch.nn as nn
import torch.nn.functional as F

noiseFeature=100


class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.fc1=nn.Linear(100,256)
        self.fc1_bn=nn.BatchNorm1d(256)
        self.fc2=nn.Linear(10,256)
        self.fc2_bn=nn.BatchNorm1d(256)
        self.fc_1=nn.Linear(512,512)
        self.fc_1_bn=nn.BatchNorm1d(512)
        self.fc_2=nn.Linear(512,1024)
        self.fc_2_bn=nn.BatchNorm1d(1024)
        self.fc_3=nn.Linear(1024,784)

    def forward(self,x,y):
        x=F.relu(self.fc1_bn(self.fc1(x)))
        y=F.relu(self.fc2_bn(self.fc2(y)))
        x=torch.cat([x,y],dim=1)
        x=F.relu(self.fc_1_bn(self.fc_1(x)))
        x=F.relu(self.fc_2_bn(self.fc_2(x)))
        x=F.tanh(self.fc_3(x))
        return x.view(-1,1,28,28)


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.fc1=nn.Linear(784,1024)
        self.fc1_bn=nn.BatchNorm1d(1024)
        self.fc_1=nn.Linear(1024,512)
        self.fc_1_bn=nn.BatchNorm1d(512)
        self.fc_2=nn.Linear(512,256)
        self.fc_2_bn=nn.BatchNorm1d(256)
        self.fc_3=nn.Linear(256,10)
    
    def forward(self,x):
        x=x.view(-1,28*28)
        x=F.leaky_relu(self.fc1(x),0.2)
        x=F.leaky_relu(self.fc_1_bn(self.fc_1(x)),0.2)
        x=F.leaky_relu(self.fc_2_bn(self.fc_2(x)),0.2)
        x=F.sigmoid(self.fc_3(x))
        return x


class classifier(nn.Module):
    def __init__(self):
        super(classifier,self).__init__()
        self.fc1=nn.Linear(784,1024)
        self.fc1_bn=nn.BatchNorm1d(1024)
        self.fc_1=nn.Linear(1024,512)
        self.fc_1_bn=nn.BatchNorm1d(512)
        self.fc_2=nn.Linear(512,256)
        self.fc_2_bn=nn.BatchNorm1d(256)
        self.fc_3=nn.Linear(256,10)
    
    def forward(self,x):
        x=x.view(-1,28*28)
        x=F.leaky_relu(self.fc1(x),0.2)
        x=F.leaky_relu(self.fc_1_bn(self.fc_1(x)),0.2)
        x=F.leaky_relu(self.fc_2_bn(self.fc_2(x)),0.2)
        x=F.sigmoid(self.fc_3(x))
        return x
