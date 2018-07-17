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
 

noiseFeature=100
#设置训练参数
batchSize=100
trainEpoch=100

def Test(dModel,cModel,testLoader):
    correct=0
    cCorrect=0
    nums=0
    for x,y in testLoader:
        x,y=Variable(x.cuda()),Variable(y.cuda())
        outputs=dModel(x)
        results=cModel(x)
        value,index=torch.max(outputs,-1)
        cValue,cIndex=torch.max(results,-1)
        correct+=torch.sum(torch.eq(index,y)).data[0]
        cCorrect+=torch.sum(torch.eq(cIndex,y)).data[0]
        nums+=len(index.data)
    print("correct",correct,"nums",nums,"acc: ",str(correct/nums*100)+"%","cAcc: ",str(cCorrect/nums*100)+"%")
    return correct/nums*100,cCorrect/nums*100


def WriteLoss(disLosses,genLosses,cLosses,acc,cAcc):
    disF=open("disLoss.txt","a")
    genF=open("genLoss.txt","a")
    cF=open("cLoss.txt","a")
    accF=open("acc.txt","a")
    cAccF=open("cAcc.txt","a")
    disF.write(str(torch.mean(torch.FloatTensor(disLosses))))
    disF.write("\n")
    genF.write(str(torch.mean(torch.FloatTensor(genLosses))))
    genF.write("\n")
    cF.write(str(torch.mean(torch.FloatTensor(cLosses))))
    cF.write("\n")
    accF.write(str(acc))
    accF.write("\n")
    cAccF.write(str(cAcc))
    cAccF.write("\n")
    disF.close()
    genF.close()
    accF.close()
    cAccF.close()



#设置图像预处理，ToTensor表示将数据转成tensor类型并且将数值归一到[0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


#读取MNIST数据
trainLoader=torch.utils.data.DataLoader(
        datasets.MNIST('data',train=True,download=True,transform=transform),batch_size=batchSize,shuffle=True)

testLoader=torch.utils.data.DataLoader(
        datasets.MNIST('data',train=False,download=True,transform=transform),batch_size=batchSize,shuffle=True)

#初始化model
Generator=generator().cuda()
Discriminator=discriminator().cuda()
Classifier=classifier().cuda()

#初始化loss
BCELoss=nn.BCELoss()
MSELoss=nn.MSELoss()
#初始化优化器
genOptim=optim.Adam(Generator.parameters(),lr=1e-4,betas=[0.5,0.99])
disOptim=optim.Adam(Discriminator.parameters(),lr=1e-4,betas=[0.5,0.99])
cOptim=optim.Adam(Classifier.parameters(),lr=1e-4,betas=[0.5,0.99])

#训练
for epoch in range(trainEpoch):
    #设置两个model为训练状态
    Generator.train()
    Discriminator.train()
    #两数组保存两个loss
    disLosses=[]
    genLosses=[]
    cLosses=[]
    
    for x,y in trainLoader:

        #设置yReal全1和yFake全0作为label
        yFake=torch.zeros(batchSize,10)
        
        #onehot y
        y=y.view(-1,1)
        onehotY=torch.zeros(batchSize,10).scatter_(1,y,1)
        #onehotY=onehotY.type(torch.FloatTensor)

        #将变量放入gpu
        x,onehotY,yFake=Variable(x.cuda()),Variable(onehotY.cuda()),Variable(yFake.cuda())

        
        #训练判别器
        disOptim.zero_grad()
        randSeed=np.random.rand(batchSize,noiseFeature)
        randSeed=Variable(torch.FloatTensor(randSeed).cuda())#获得随机种子
        fakeImg=Generator(randSeed,onehotY)#生成fake图片
        disFakeOutput=Discriminator(fakeImg)#fake图片放入判别器
        fakeLoss=MSELoss(disFakeOutput,yFake)


        realImg=x
        disRealOutput=Discriminator(realImg)#real图片放入判别器
        realLoss=MSELoss(disRealOutput,onehotY)
        disLoss=(realLoss+fakeLoss)/2#将两个loss相加
        disLoss.backward()
        disLosses.append(disLoss.data[0])
        disOptim.step()
        
        #训练生成器
        genOptim.zero_grad()
        randSeed=np.random.rand(batchSize,noiseFeature)
        randSeed=Variable(torch.FloatTensor(randSeed).cuda())
        fakeImg=Generator(randSeed,onehotY)
        
        disFakeOutput=Discriminator(fakeImg)#生成fake图片
        genLoss=MSELoss(disFakeOutput,onehotY)
        genLoss.backward()
        genLosses.append(genLoss.data[0])
        genOptim.step()



        Classifier.zero_grad()
        results=Classifier(x)
        cLoss=MSELoss(results,onehotY)
        cLoss.backward()
        cLosses.append(cLoss.data[0])
        cOptim.step()


    Generator.eval()
    Discriminator.eval()

    #生成测试图像
    randSeedTest=np.random.rand(100,noiseFeature)
    randSeedTest=Variable(torch.FloatTensor(randSeedTest).cuda())
    ids=[]
    for i in range(10):
        for j in range(10):
            ids.append(i)
    ids=torch.LongTensor(ids).view(-1,1)
    testY=torch.zeros(100,10).scatter_(1,ids,1)
    testY=Variable(testY.cuda())
    testImg=Generator(randSeedTest,testY)
    testImg=testImg.view(-1,1,28,28).data/2+0.5
    save_image(testImg,f"demo/demo_Epoch{epoch}.jpg",nrow=10)

    print("epoch%d, disLoss:%.3f, genLoss:%.3f, cLoss:%.3f"%(epoch,torch.mean(torch.FloatTensor(disLosses)),torch.mean(torch.FloatTensor(genLosses)),torch.mean(torch.FloatTensor(cLosses))))


    acc,cAcc=Test(Discriminator,Classifier,testLoader)
    WriteLoss(disLosses,genLosses,cLosses,acc,cAcc)


