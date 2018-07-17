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
import itertools


def WriteLoss(disLosses,genLosses,infoLosses):
    disF=open("disLoss.txt","a")
    genF=open("genLoss.txt","a")
    infoF=open("infoLoss.txt","a")
    disF.write(str(torch.mean(torch.FloatTensor(disLosses))))
    disF.write("\n")
    genF.write(str(torch.mean(torch.FloatTensor(genLosses))))
    genF.write("\n")
    infoF.write(str(torch.mean(torch.FloatTensor(infoLosses))))
    infoF.write("\n")
    disF.close()
    genF.close()
    infoF.close()

noiseFeature=60
#设置训练参数
batchSize=100
trainEpoch=100

#设置图像预处理，ToTensor表示将数据转成tensor类型并且将数值归一到[0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


#读取MNIST数据
trainLoader=torch.utils.data.DataLoader(
        datasets.MNIST('data',train=True,download=True,transform=transform),batch_size=batchSize,shuffle=True)

#初始化model
Generator=generator().cuda()
Discriminator=discriminator().cuda()

#初始化loss
BCELoss=nn.BCELoss()
MSELoss=nn.MSELoss()

#初始化优化器
genOptim=optim.Adam(Generator.parameters(),lr=1e-4,betas=[0.5,0.99])
disOptim=optim.Adam(Discriminator.parameters(),lr=1e-4,betas=[0.5,0.99])
infoOptim=optim.Adam(itertools.chain(Generator.parameters(), Discriminator.parameters()), lr=1e-4,betas=[0.5,0.99])
#infoLoss同时传递给生成器和判别器

#训练
for epoch in range(trainEpoch):
    #设置两个model为训练状态
    Generator.train()
    Discriminator.train()
    #两数组保存两个loss
    disLosses=[]
    genLosses=[]
    infoLosses=[]
    for x,y in trainLoader:
        #设置yReal全1和yFake全0作为label
        yReal=torch.ones(batchSize,1)
        yFake=torch.zeros(batchSize,1)
        
        #onehot y
        y=y.view(-1,1)
        onehotY=torch.zeros(batchSize,10).scatter_(1,y,1)
        #onehotY=onehotY.type(torch.FloatTensor)

        #将变量放入gpu
        x,onehotY,yReal,yFake=Variable(x.cuda()),Variable(onehotY.cuda()),Variable(yReal.cuda()),Variable(yFake.cuda())

        
        #训练判别器
        disOptim.zero_grad()
        randSeed=np.random.rand(batchSize,noiseFeature)
        randSeed=Variable(torch.FloatTensor(randSeed).cuda())#获得随机种子
        randC2=Variable(torch.FloatTensor(np.random.rand(batchSize,1)*2-1).cuda()).view(-1,1)#生成随机连续变量c2
        randC3=Variable(torch.FloatTensor(np.random.rand(batchSize,1)*2-1).cuda()).view(-1,1)#生成随机连续变量c3
        fakeImg=Generator(randSeed,onehotY,randC2,randC3)#生成fake图片
        disFakeOutput,c1,c2,c3=Discriminator(fakeImg)#fake图片放入判别器
        fakeLoss=BCELoss(disFakeOutput,yFake)


        realImg=x
        disRealOutput,_,_,_=Discriminator(realImg)#real图片放入判别器
        realLoss=BCELoss(disRealOutput,yReal)
        disLoss=(realLoss+fakeLoss)/2#将两个loss相加
        disLoss.backward()
        disLosses.append(disLoss.data[0])
        disOptim.step()
        
        #训练生成器
        genOptim.zero_grad()
        randSeed=np.random.rand(batchSize,noiseFeature)
        randSeed=Variable(torch.FloatTensor(randSeed).cuda())
        randC2=Variable(torch.FloatTensor(np.random.rand(batchSize,1)*2-1).cuda()).view(-1,1)#生成随机连续变量c2
        randC3=Variable(torch.FloatTensor(np.random.rand(batchSize,1)*2-1).cuda()).view(-1,1)#生成随机连续变量c3
        fakeImg=Generator(randSeed,onehotY,randC2,randC3)
        
        disFakeOutput,c1,c2,c3=Discriminator(fakeImg)#生成fake图片
        genLoss=BCELoss(disFakeOutput,yReal)
        genLoss.backward(retain_graph=True)
        genLosses.append(genLoss.data[0])
        genOptim.step()



        #互信息同时训练D和G
        disOptim.zero_grad()
        Generator.zero_grad()
        c1Loss=BCELoss(c1,onehotY)#类别和onehot的label
        c2Loss=MSELoss(c2,randC2)#输入c2和输出c2
        c3Loss=MSELoss(c3,randC3)#输入c3和输出c3
        infoLoss=c1Loss+c2Loss+c3Loss
        infoLoss.backward()
        infoLosses.append(infoLoss.data[0])
        infoOptim.step()


    Generator.eval()
    Discriminator.eval()

    #生成测试图像1(控制c3)
    randSeedTest=np.random.rand(batchSize,noiseFeature)
    randSeedTest=Variable(torch.FloatTensor(randSeedTest).cuda())
    c1=[]
    c2=[]
    c3=[]
    for i in range(10):
        for j in range(10):
            c1.append(i)
            c2.append((j-5)/5.0)
            c3.append(0)
    c1=torch.LongTensor(c1).view(-1,1)
    c2=torch.FloatTensor(c2).view(-1,1)
    c3=torch.FloatTensor(c3).view(-1,1)
    testY=torch.zeros(100,10).scatter_(1,c1,1)
    testY=Variable(testY.cuda())
    c2=Variable(c2.cuda())
    c3=Variable(c3.cuda())
    #生成保存生成图像
    testImg=Generator(randSeedTest,testY,c2,c3)
    testImg=testImg.view(-1,1,28,28).data/2+0.5
    save_image(testImg,f"demo/c2/demo_Epoch{epoch}_c2.jpg",nrow=10)


    #生成测试图像2(控制c2)
    randSeedTest=np.random.rand(batchSize,noiseFeature)
    randSeedTest=Variable(torch.FloatTensor(randSeedTest).cuda())
    c1=[]
    c2=[]
    c3=[]
    for i in range(10):
        for j in range(10):
            c1.append(i)
            c2.append(0)
            c3.append((j-5)/5.0)
    c1=torch.LongTensor(c1).view(-1,1)
    c2=torch.FloatTensor(c2).view(-1,1)
    c3=torch.FloatTensor(c3).view(-1,1)
    testY=torch.zeros(100,10).scatter_(1,c1,1)
    testY=Variable(testY.cuda())
    c2=Variable(c2.cuda())
    c3=Variable(c3.cuda())
    #生成保存生成图像
    testImg=Generator(randSeedTest,testY,c2,c3)
    testImg=testImg.view(-1,1,28,28).data/2+0.5
    save_image(testImg,f"demo/c3/demo_Epoch{epoch}_c3.jpg",nrow=10)
    
    #输出loss
    WriteLoss(disLosses,genLosses,infoLosses)
    print("epoch%d, disLoss:%.3f, genLoss:%.3f , infoLoss:%.3f"%(epoch,torch.mean(torch.FloatTensor(disLosses)),torch.mean(torch.FloatTensor(genLosses)),torch.mean(torch.FloatTensor(infoLosses))))







