import os
import torch
import torch.nn as nn
import torch.nn.functional as F

noiseFeature=60

#生成器和判别器都使用卷积网络
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.fc1=nn.Linear(60+12,1024)#60为随机噪声，10为类别，2为连续随机变量。
        self.fc1_bn=nn.BatchNorm1d(1024)
        self.fc2=nn.Linear(1024,128*7*7)
        self.fc2_bn=nn.BatchNorm1d(128*7*7)
        self.deconv1=nn.ConvTranspose2d(128,64,4,2,padding=1)
        self.deconv1_bn=nn.BatchNorm2d(64)
        self.deconv2=nn.ConvTranspose2d(64,1,4,2,padding=1)

    def forward(self,x,c1,c2,c3):#c1表示类别，c2，c3为两个连续随机变量
        x=torch.cat([x,c1,c2,c3],dim=1)
        x=F.relu(self.fc1_bn(self.fc1(x)))
        x=F.relu(self.fc2_bn(self.fc2(x)))
        x=x.view(-1,128,7,7)
        x=self.deconv1_bn(self.deconv1(x))
        x=self.deconv2(x)
        x=F.tanh(x)
        return x.view(-1,1,28,28)


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 13),
            nn.Sigmoid(),
        )
    
    def forward(self,x):
        x=self.conv(x)
        x=x.view(-1,128*7*7)
        x=self.fc(x)
	#结果分成4个部分，dis表示是否为生成图像，c1表示类别，c2，c3为两个连续随机变量。
        dis=x[:,0:1]#disx[:,0]会报错，因为得到size是[batchSize]，我们需要的是[batchSize,1]
        c1=x[:,1:11]
        c2=x[:,11:12]
        c3=x[:,12:13]
        return dis,c1,c2,c3
