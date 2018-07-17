import os
import torch
import torch.nn as nn
import torch.nn.functional as F

noiseFeature=100


def deconv_relu_bn(inChannel,outChannel,kernelSize,stride,padding):
    return nn.Sequential(nn.ConvTranspose2d(inChannel,outChannel,kernelSize,stride,padding=padding),
                         nn.ReLU(),
                         nn.BatchNorm2d(outChannel))


def conv_lrelu_bn(inChannel,outChannel,kernelSize,stride,padding):
    return nn.Sequential(nn.Conv2d(inChannel,outChannel,kernelSize,stride,padding=padding),
                         nn.LeakyReLU(0.2),
                         nn.BatchNorm2d(outChannel))

class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.fc1=nn.Linear(100,4*4*256)
        self.conv1=deconv_relu_bn(256,64,4,2,padding=1)
        self.conv2=deconv_relu_bn(64,32,4,2,padding=1)
        self.conv3=nn.ConvTranspose2d(32,3,4,2,padding=1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=x.view(-1,256,4,4)
        x=self.conv1(x)
        x=self.conv2(x)
        x=F.tanh(self.conv3(x))
        return x


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.conv1=conv_lrelu_bn(3,64,5,2,padding=2)
        self.conv2=conv_lrelu_bn(64,128,3,2,padding=1)
        self.conv3=conv_lrelu_bn(128,256,3,2,padding=1)
        self.fc1=nn.Linear(4*4*256,1)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(-1,256*4*4)
        x=F.sigmoid(self.fc1(x))
        return x
