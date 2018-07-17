import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable
from model import *
import cv2
import numpy as np
from torchvision.utils import save_image
 
 
 
def WriteLoss(disLosses,genLosses):
    disF=open("disLoss.txt","a")
    genF=open("genLoss.txt","a")
    disF.write(str(torch.mean(torch.FloatTensor(disLosses))))
    disF.write("\n")
    genF.write(str(torch.mean(torch.FloatTensor(genLosses))))
    genF.write("\n")
    disF.close()
    genF.close()
 
noiseFeature=30
#设置训练参数
batchSize=100
trainEpoch=100

#设置图像预处理，ToTensor表示将数据转成tensor类型并且将数值归一到[0,1]
transform = transforms.Compose([
    transforms.ToTensor()
])


#读取MNIST数据
trainLoader=torch.utils.data.DataLoader(
        datasets.MNIST('data',train=True,download=True,transform=transform),batch_size=batchSize,shuffle=True)

#初始化model
Generator=generator().cuda()
Discriminator=discriminator().cuda()

#初始化loss
BCELoss=nn.BCELoss()

#初始化优化器
genOptim=optim.Adam(Generator.parameters(),lr=1e-4,betas=[0.5,0.99])
disOptim=optim.Adam(Discriminator.parameters(),lr=1e-4,betas=[0.5,0.99])

#训练
for epoch in range(trainEpoch):
    #设置两个model为训练状态
    Generator.train()
    Discriminator.train()
    #两数组保存两个loss
    disLosses=[]
    genLosses=[]
    
    for x,_ in trainLoader:
        #将x即图片平铺
        x=x.view(-1,28*28)
        #设置yReal全1和yFake全0作为label
        yReal=torch.ones(batchSize,1)
        yFake=torch.zeros(batchSize,1)
        #将变量放入gpu
        x,yReal,yFake=Variable(x.cuda()),Variable(yReal.cuda()),Variable(yFake.cuda())

        #训练判别器
        disOptim.zero_grad()
        randSeed=np.random.rand(batchSize,noiseFeature)*2-1
        randSeed=Variable(torch.FloatTensor(randSeed).cuda())#获得随机种子
        fakeImg=Generator(randSeed)#生成fake图片
        disFakeOutput=Discriminator(fakeImg)#fake图片放入判别器
        fakeLoss=BCELoss(disFakeOutput,yFake)


        realImg=x
        disRealOutput=Discriminator(realImg)#real图片放入判别器
        realLoss=BCELoss(disRealOutput,yReal)
        disLoss=(realLoss+fakeLoss)/2#将两个loss相加
        disLoss.backward()
        disLosses.append(disLoss.data[0])
        disOptim.step()
        
        #训练生成器
        genOptim.zero_grad()
        randSeed=np.random.rand(batchSize,noiseFeature)*2-1
        randSeed=Variable(torch.FloatTensor(randSeed).cuda())
        fakeImg=Generator(randSeed)
        
        disFakeOutput=Discriminator(fakeImg)#生成fake图片
        genLoss=BCELoss(disFakeOutput,yReal)
        genLoss.backward()
        genLosses.append(genLoss.data[0])
        genOptim.step()

    Generator.eval()
    Discriminator.eval()

    #生成测试图像
    randSeedTest=np.random.rand(25,noiseFeature)*2-1
    randSeedTest=Variable(torch.FloatTensor(randSeedTest).cuda())

    testImg=Generator(randSeedTest)
    testImg=testImg.view(-1,1,28,28).data
    save_image(testImg,f"demo/demo_Epoch{epoch}.jpg",nrow=5)
    WriteLoss(disLosses,genLosses)
    print("epoch%d, disLoss:%.3f, genLoss:%.3f"%(epoch,torch.mean(torch.FloatTensor(disLosses)),torch.mean(torch.FloatTensor(genLosses))))







