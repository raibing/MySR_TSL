import  numpy as np
import torch as tor
from  torch import nn
class skip_clipLSTM(nn.Module):
    def __init__(self,dd,input_dim1,hidden_dim1,nlayer1,hidden_dim2,nlayer2,hidden_dim3,nlayer3):
        super(skip_clipLSTM, self).__init__()

        self.lstm1=nn.LSTM(input_dim1,hidden_dim1,nlayer1)
        self.lstm2=nn.LSTM(hidden_dim1,hidden_dim2,nlayer2)
        self.lstm3=nn.LSTM(hidden_dim2,hidden_dim3,nlayer3)
        self.lastH2 = self.initHiddenstate(nlayer2,dd,hidden_dim2)
        self.lastH3 = self.initHiddenstate(nlayer3,dd,hidden_dim3)
        self.lastC2 = self.initCHiddenstate(nlayer2,dd,hidden_dim2)
        self.lastC3 = self.initCHiddenstate(nlayer3,dd,hidden_dim3)

    def toDevice(self,dev):
        self.lastH2.to(dev)
        self.lastH3.to(dev)
        self.lastC2.to(dev)
        self.lastC3.to(dev)

    def forward(self, input,lasth,lastc):
        out1,(h1,c1)=self.lstm1(input,(lasth,lastc))
        x1=tor.cat((lasth,h1))



        out2,(h2,c2) = self.lstm2(x1,(self.lastH2,self.lastC2))
        x2 = tor.cat(( self.lastH2,h2))
        self.lastH2 = h2
        self.lastC2=c2

        out3,(h3,c3)=self.lstm3(x2,(self.lastH3,self.lastC3))
        self.lastH3 = h3.view(h3.size())
        self.lastC3=c3

        return  out3,(h3,c3)


    def initHiddenstate(self,n,t,h):
        return tor.zeros(n,t,h)

    def initCHiddenstate(self,n,t,h):
        return tor.zeros(n, t, h)
class LearningClassier(nn.Module):
    def __init__(self,indim,NumberLabel,drop=0.5):
        super(LearningClassier,self).__init__()
        # fc
        self.NumberLabel=NumberLabel
        self.linear3 = nn.Linear(indim, NumberLabel)


        self.dropout = nn.Dropout(p=drop)

        self.linear4 = nn.Linear(indim, 1)



    def forward(self, H):
        size=H.size()
        N=size[0]
        Po= tor.FloatTensor(N, self.NumberLabel)
        for i in range (N):
            Os = self.linear3(H[i])
            # tmp numberlabel*1 *scLstmDim
            Os = Os.permute(2, 1, 0)
            tmp = Os.view(Os.size())
            # drop between fc3 fc4
            tmp = self.dropout(tmp)

            tmp = self.linear4(tmp)

            # tmp1 numberlabel*1 *1
            tmp = tmp.permute(2, 1, 0)
            tmp1 = tmp.view(tmp.size())
            # pro numberlabel
            probability = tmp1[0][0].view(self.NumberLabel)
            ## feed sotfmax
            probability = probability.softmax(0)
        # Po N*numberlabel
            Po[i] = probability
        return Po