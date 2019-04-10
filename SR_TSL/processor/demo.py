import  numpy as np

import  torch as tor
from torch import  nn
from torch.autograd import Variable
from  SR_TSL.net.sr import grnn
from SR_TSL.processor import  myIO
from  SR_TSL.net import tsl
from SR_TSL.net import sr
from torch.optim import Adam
from SR_TSL.utils.tool import expendRank
import random
import math
def OnlyVec():
    path = "Data\\train\\not\\"

    # data = myIO.readOp3d(path)


    k = 67
    b = 8
    b2 = 8
    b3 = 8
    b4 = 16
    b5 = b3
    b6 = 12
    b7 = 16

    dim = 4
    clipM = 10
    d = 5
    d1 = 8
    d2 = 6
    d3 = d1
    NumberLabel = 5

    weight = tor.ones(k, b)
    bias = tor.zeros(k, b)
    #test data
    data = tor.rand(11, 67, 4)
    [n, kkkkkk, bbbbb] = data.size()
    ##net init
    # fc 1
    linear = nn.Linear(dim, b)
    # grnn
    net = grnn(weights=weight, bias=bias, epoch=5)
    # fc2
    linear2 = nn.Linear(b, b2)
    # skip clip
    skLSTM = tsl.skip_clipLSTM(1, b2, b3, d1, b4, d2, b5, d3)
    # fc3 F01
    linear3 = nn.Linear(b2, NumberLabel)
    # fc 4 F02
    linear4 = nn.Linear(b2, 1)

    N = math.floor(n / clipM)
    lastH = tor.zeros(d1, 1, b2)
    lastC = tor.zeros(d1, 1, b2)
    Hm = tor.zeros(d1, 1, b2)
    Hc = tor.zeros(d1, 1, b2)
    lastq=tor.zeros(k,b2)
    for i in range(N):
        print("clip ", i, " start")
        Q = tor.FloatTensor(k, 1, b2)

        for j in range(clipM):
            o = linear(data[i * clipM + j])
            # grnn
            output = net(o)
            ## fc 2
            ##q 67* 8 --> k*b2
            q = linear2(output)

            ## q->inn k*1*b2
            ## q- lastq = vector
            inn = expendRank(1, q-lastq)
            lastq = q
            ## skip-clip lstm for one frame
            outt, (lastH, lastC) = skLSTM(inn, lastH, lastC)

            # print("frame ", i*clipM+j, " finished")

        Hm = tor.add(lastH, Hm)
        Hc = tor.add(lastC, Hc)
        lastH = Hm
        lastC = Hc
        # hm 8*1*8  d1*1*b2
        ## d3==b2  b5==b3
        # Om b2*1*numberlabel
        Om = linear3(Hm)
        # tmp numberlabel*1 *b2
        tmp = Om.permute(2, 1, 0)
        tmp1 = linear4(tmp)
        # tmp1 numberlabel*1 *1
        tmp1 = tmp1.permute(2, 1, 0)
        # p pro numberlabel
        probability = tmp1[0][0]
        probability = probability.softmax(0)
        print("clip ", i, " finished")
        print(probability)

def OnlyPostiton():
    path = "Data\\train\\not\\"
    data = myIO.readOp3d(path)
    k = 67
    b = 8
    b2 = 8
    b3 = 8
    b4 = 16
    b5 = b3
    b6 = 12
    b7 = 16

    dim = 4
    clipM = 10
    d = 5
    d1 = 8
    d2 = 6
    d3 = d1
    NumberLabel = 5
    weight = tor.ones(k, b)
    bias = tor.zeros(k, b)
    data = tor.rand(11, 67, 4)
    [n, kkkkkk, bbbbb] = data.size()
    ##net init
    # fc 1
    linear = nn.Linear(dim, b)
    # grnn
    net = grnn(weights=weight, bias=bias, epoch=5)
    # fc2
    linear2 = nn.Linear(b, b2)
    # skip clip
    skLSTM = tsl.skip_clipLSTM(1, b2, b3, d1, b4, d2, b5, d3)
    # fc3 F01
    linear3 = nn.Linear(b2, NumberLabel)
    # fc 4 F02
    linear4 = nn.Linear(b2, 1)

    N = math.floor(n / clipM)
    lastH = tor.zeros(d1, 1, b2)
    lastC = tor.zeros(d1, 1, b2)
    Hm = tor.zeros(d1, 1, b2)
    Hc = tor.zeros(d1, 1, b2)
    for i in range(N):
        print("clip ", i, " start")
        Q = tor.FloatTensor(k, 1, b2)

        for j in range(clipM):
            o = linear(data[i * clipM + j])
            # grnn
            output = net(o)
            ## fc 2
            ##q 67* 8 --> k*b2
            q = linear2(output)
            ## q->inn k*1*b2
            inn = expendRank(1, q)
            ## skip-clip lstm for one frame
            outt, (lastH, lastC) = skLSTM(inn, lastH, lastC)

            # print("frame ", i*clipM+j, " finished")

        Hm = tor.add(lastH, Hm)
        Hc = tor.add(lastC, Hc)
        lastH = Hm
        lastC = Hc
        # hm 8*1*8  d1*1*b2
        ## d3==b2  b5==b3
        # Om b2*1*numberlabel
        Om = linear3(Hm)
        # tmp numberlabel*1 *b2
        tmp = Om.permute(2, 1, 0)
        tmp1 = linear4(tmp)
        # tmp1 numberlabel*1 *1
        tmp1 = tmp1.permute(2, 1, 0)
        # p pro numberlabel
        probability = tmp1[0][0]
        probability = probability.softmax(0)
        print("clip ", i, " finished")
        print(probability)

def bothPV():
    path = "Data\\train\\tik\\"

    data = myIO.readOp3d(path)

    k = 67
    b = 8
    b2 = 8
    b3 = 8
    b4 = 16
    b5 = b3


    dim = 4
    clipM = 10
    d = 5
    d1 = 8
    d2 = 6
    d3 = d1
    NumberLabel = 5

    weight = tor.ones(k, b)
    bias = tor.zeros(k, b)

    [n, kkkkkk, bbbbb] = data.size()
    ##net init
    # fc 1
    linear = nn.Linear(dim, b)
    # grnn
    net = grnn(weights=weight, bias=bias, epoch=5)
    # fc2
    linear2 = nn.Linear(b, b2)
    # skip clip
    skLSTM_Vec = tsl.skip_clipLSTM(1, b2, b3, d1, b4, d2, b5, d3)
    skLSTM_Pos = tsl.skip_clipLSTM(1, b2, b3, d1, b4, d2, b5, d3)
    # fc3 F01
    linear3 = nn.Linear(b2, NumberLabel)

    # fc 4 F02
    linear4 = nn.Linear(b2, 1)

    N = math.floor(n / clipM)
    lastHV = tor.zeros(d1, 1, b2)
    lastCV = tor.zeros(d1, 1, b2)
    HmV = tor.zeros(d1, 1, b2)
    HcV = tor.zeros(d1, 1, b2)

    lastHP = tor.zeros(d1, 1, b2)
    lastCP = tor.zeros(d1, 1, b2)
    HmP = tor.zeros(d1, 1, b2)
    HcP = tor.zeros(d1, 1, b2)

    lastq = tor.zeros(k, b2)
    for i in range(N):
        print("clip ", i, " start")
        Q = tor.FloatTensor(k, 1, b2)

        for j in range(clipM):
            o = linear(data[i * clipM + j])
            # grnn
            output = net(o)
            ## fc 2
            ##q 67* 8 --> k*b2
            q = linear2(output)

            ## q->inn k*1*b2
            ## q- lastq = vector
            innV = expendRank(1, q - lastq)
            innP = expendRank(1, q)
            lastq = q
            ## skip-clip lstm for one frame
            outtV, (lastHV, lastCV) = skLSTM_Vec(innV, lastHV, lastCV)
            outtP, (lastHP, lastCP) = skLSTM_Pos(innP, lastHP, lastCP)

            # print("frame ", i*clipM+j, " finished")

        HmV = tor.add(lastHV, HmV)
        HcV = tor.add(lastCV, HcV)
        lastHV = HmV
        lastCV = HcV
        HmP = tor.add(lastHP, HmP)
        HcP = tor.add(lastCP, HcP)
        lastHP = HmP
        lastCP = HcP

        # Hs 8*1*8  d1*1*b2
        ## d3==b2  b5==b3
        # Om b2*1*numberlabel
        Hs = tor.add(HmP, HmV)
        Os = linear3(Hs)
        # tmp numberlabel*1 *b2
        tmp = Os.permute(2, 1, 0)
        tmp1 = linear4(tmp)
        # tmp1 numberlabel*1 *1
        tmp1 = tmp1.permute(2, 1, 0)
        # p pro numberlabel
        probability = tmp1[0][0]
        ## feed sotfmax
        probability = probability.softmax(0)
        print("clip ", i, " finished")
        print(probability)

def test():
    def initLabelMatrix(n, x):
        matrix = tor.zeros(n)
        matrix[x] = 1
        return matrix

    k = 67
    grnndim = 8
    hm = 4
    n = 6
    weight = tor.ones(k, grnndim)
    bias = tor.zeros(k, grnndim)
    cri = sr.myLoss()
    rnn = sr.sr_tsl(8, n)
    learn = tsl.LearningClassier(8, n, drop=0.5)
    # lst=nn.LSTM(hm,hm,hm)
    # rnn=RNN()
    lasth = tor.zeros(4, 1, hm)
    lasth2 = tor.zeros(4, 1, hm)
    lasth3 = tor.zeros(4, 1, hm)
    lastc = tor.zeros(4, 1, hm)
    lastc1 = tor.zeros(4, 1, hm)
    lastc2 = tor.zeros(4, 1, hm)
    optimizer = Adam(learn.parameters(), lr=0.001)

    for i in range(3):
        data = tor.rand(31, k, 4)
        loss = 0
        hp, hc = rnn(data)
        hp = hp.detach()

        # hp=Variable(hp,requires_grad=False)
        # hp=tor.rand(2,8,1,8)


        out = learn(hp)

        x = math.floor((random.random() * 100) % n)
        ytrue = initLabelMatrix(n, x)

        loss = cri(out, ytrue)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # for p in rnn.parameters():
        #   p.data.add_(-0.001, p.grad.data)
        print(i, ":term")
