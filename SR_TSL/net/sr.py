import numpy as np

import tensorflow as tf
import  torch as tor
import  torch.nn  as nn
import torch.nn.functional as functional
import math
from torch.autograd import Variable
from SR_TSL.utils.tool import expendRank
from SR_TSL.net import tsl
from pathlib import Path
from  SR_TSL.processor import myIO
class Model:
    def __init__(self):

        self.k = 67
        self.b = 8
        self.b2 = 8
        self.b3 = 8
        self.b4 = 16
        self.b5 = self.b3
        self.dim = 4
        self.clipM = 10
        self.d = 5
        self.d1 = 8
        self.d2 = 6
        self.d3 = self.d1
        self.NumberLabel = 5
        self.ep=5
        self.learning_rate=0.005
        self.vecinit()

    def vecinit(self):
        self.weight = tor.ones(self.k, self.b)
        self.bias = tor.zeros(self.k, self.b)
        self.lastHV = tor.zeros(self.d1, 1, self.b2)
        self.lastCV = tor.zeros(self.d1, 1, self.b2)
        self.HmV = tor.zeros(self.d1, 1, self.b2)
        self.HcV = tor.zeros(self.d1, 1, self.b2)

        self.lastHP = tor.zeros(self.d1, 1, self.b2)
        self.lastCP = tor.zeros(self.d1, 1, self.b2)
        self.HmP = tor.zeros(self.d1, 1, self.b2)
        self.HcP = tor.zeros(self.d1, 1, self.b2)

        self.lastq = tor.zeros(self.k, self.b2)
        self.criterion = myLoss()


    def netinit(self):


        ##net init
        # fc 1
        self.linear = nn.Linear(self.dim, self.b)
        # grnn
        self. net = grnn(weights=self.weight, bias=self.bias, epoch=5)
        # fc2
        self.linear2 = nn.Linear(self.b, self.b2)
        # skip clip
        self.skLSTM_Vec = tsl.skip_clipLSTM(1, self.b2, self.b3, self.d1, self.b4, self.d2, self.b5, self.d3)
        self.skLSTM_Pos = tsl.skip_clipLSTM(1, self.b2, self.b3, self.d1, self.b4, self.d2, self.b5, self.d3)
        # fc3 F01
        self.linear3 = nn.Linear(self.b2, self.NumberLabel)

        # fc 4 F02
        self.linear4 = nn.Linear(self.b2, 1)








    def setLabels(self,files):
        labels = []
        Classes = {}
        c = 0
        for lab in files.glob('*'):
            labels.append(lab.name)
            Classes[lab]=c
            c = c + 1
        self.NumberLabel = len(labels)
        self.Classes=Classes
        self.labels=labels

    def initLabelMatrix(self,label):
        matrix=tor.zeros(self.NumberLabel)
        matrix[self.Classes[label]]=1
        return  matrix

    def train(self,train_path):
        files = Path(train_path)
        self.setLabels(files)
        ls=0
        ee=2
        for label in files.glob('*'):
            print("label name:", label.name)

            if ee<0:
                break
            ee = ee - 1
            for sample in label.glob('*'):
                    print("smaples name:", sample.name)

                    data=myIO.readOp3d(sample)
                    n = data.size()
                    N = math.floor(n[0] / self.clipM)
                    for i in range(N):
                        print("clip ", i, " start")
                        # Q = tor.FloatTensor(self.k, 1, self.b2)

                        for j in range(self.clipM):
                            o = self.linear(data[i * self.clipM + j])
                            # grnn
                            output = self.net(o)
                            ## fc 2
                            ##q 67* 8 --> k*b2
                            q = self.linear2(output)
                            del output
                            ## q->inn k*1*b2
                            ## q- lastq = vector
                            innV = expendRank(1, q - self.lastq)
                            innP = expendRank(1, q)
                            self.lastq = q
                            ## skip-clip lstm for one frame
                            outtV, (self.lastHV, self.lastCV) = self.skLSTM_Vec(innV, self.lastHV, self.lastCV)
                            outtP, (self.lastHP, self.lastCP) = self.skLSTM_Pos(innP, self.lastHP, self.lastCP)
                            del outtP,outtV

                            # print("frame ", i*clipM+j, " finished")
                        #get hmV,h
                        self.HmV = tor.add(self.lastHV, self.HmV)
                        self.HcV = tor.add(self.lastCV, self.HcV)
                        self.lastHV = self.HmV
                        self.lastCV = self.HcV
                        self.HmP = tor.add(self.lastHP, self.HmP)
                        self.HcP = tor.add(self.lastCP, self.HcP)
                        self.lastHP = self.HmP
                        self.lastCP = self.HcP

                        # Hs 8*1*8  d1*1*b2
                        ## d3==b2  b5==b3
                        # Om b2*1*numberlabel
                        Hs = tor.add(self.HmP, self.HmV)
                        Os = self.linear3(Hs)
                        # tmp numberlabel*1 *b2
                        tmp = Os.permute(2, 1, 0)
                        tmp1 = self.linear4(tmp)
                        # tmp1 numberlabel*1 *1
                        tmp1 = tmp1.permute(2, 1, 0)
                        # p pro numberlabel
                        probability = tmp1[0][0]
                        ## feed sotfmax
                        probability = probability.softmax(0)
                        ytrue=self.initLabelMatrix(label)
                        ls=ls+(i/N)*tor.sum(ytrue.mul(probability.log()),-1)
                        del Hs,Os,tmp,tmp1
                        print("clip ", i, " finished")
                        print("current loss:",-ls)
                    del data
        ls=-ls
        print("final loss:",ls)
    def train2(self,train_path):
        files = Path(train_path)
        self.setLabels(files)
        ls = 0
        ll=0
        ee = 2
        cri=nn.NLLLoss()
        for label in files.glob('*'):
            print("label name:", label.name)

            if ee < 0:
                break
            ee = ee - 1
            for sample in label.glob('*'):
                print("smaples name:", sample.name)

                data = myIO.readOp3d(sample)
                probabilities,loss=self.fit(data,label.name)
                ls+=loss

                print("loss:",ls)





    def saveNet(self,paths):

        assert  len(paths)==7
        tor.save(self.linear.state_dict(),paths[0])
        tor.save(self.net.state_dict(),paths[1])
        tor.save(self.linear2.state_dict(),paths[2])
        tor.save(self.skLSTM_Pos.state_dict(),paths[3])
        tor.save(self.skLSTM_Vec.state_dict(),paths[4])
        tor.save(self.linear3.state_dict(),paths[5])
        tor.save(self.linear4.state_dict(),paths[6])

    def loadNet(self,paths):
        assert len(paths) == 7
        ##net load form model
        # fc 1
        self.linear = nn.Linear(self.dim, self.b)
        model=tor.load(paths[0])
        self.linear.load_state_dict(model)
        self.linear.eval()
        # grnn
        self.net = grnn(weights=self.weight, bias=self.bias, epoch=self.ep)
        model = tor.load(paths[1])
        self.net.load_state_dict(model)
        self.net.eval()
        # fc2
        self.linear2 = nn.Linear(self.b, self.b2)
        model = tor.load(paths[2])
        self.linear2.load_state_dict(model)
        self.net.eval()
        # skip clip
        self.skLSTM_Vec = tsl.skip_clipLSTM(1, self.b2, self.b3, self.d1, self.hidden3, self.d2, self.b5, self.d3)
        self.skLSTM_Pos = tsl.skip_clipLSTM(1, self.b2, self.b3, self.d1, self.hidden3, self.d2, self.b5, self.d3)
        model = tor.load(paths[3])
        self.skLSTM_Pos.load_state_dict(model)
        self.skLSTM_Pos.eval()
        model = tor.load(paths[4])
        self.skLSTM_Vec.load_state_dict(model)
        self.skLSTM_Vec.eval()

        # fc3 F01
        self.linear3 = nn.Linear(self.b2, self.NumberLabel)
        model = tor.load(paths[5])
        self.linear3.load_state_dict(model)
        self.linear3.eval()

        # fc 4 F02
        self.linear4 = nn.Linear(self.b2, 1)
        model = tor.load(paths[6])
        self.linear4.load_state_dict(model)
        self.linear4.eval()
    def extractFeature(self,videodata):
        n = videodata.size()
        N = math.floor(n[0] / self.clipM)
        Q = tor.FloatTensor(N*n,self.k, self.b2)
        for i in range(N):
            print("clip ", i, " start")


            for j in range(self.clipM):
                o = self.linear(videodata[i * self.clipM + j])
                # grnn
                output = self.net(o)
                ## fc 2
                ##q 67* 8 --> k*b2
                q = self.linear2(output)
                Q[i*self.clipM+j]=q
    def fit(self,videodata,label):
        n = videodata.size()
        N = math.floor(n[0] / self.clipM)
        Po=tor.FloatTensor(N,self.NumberLabel)


        for i in range(N):
            print("clip ", i, " start")
            # Q = tor.FloatTensor(self.k, 1, self.b2)

            for j in range(self.clipM):
                o = self.linear(videodata[i * self.clipM + j])
                # grnn
                output = self.net(o)
                ## fc 2
                ##q 67* 8 --> k*b2
                q = self.linear2(output)
                del output
                ## q->inn k*1*b2
                ## q- lastq = vector
                innV = expendRank(1, q - self.lastq)
                innP = expendRank(1, q)
                self.lastq = q
                ## skip-clip lstm for one frame
                outtV, (self.lastHV, self.lastCV) = self.skLSTM_Vec(innV, self.lastHV, self.lastCV)
                outtP, (self.lastHP, self.lastCP) = self.skLSTM_Pos(innP, self.lastHP, self.lastCP)
                del outtP, outtV

                # print("frame ", i*clipM+j, " finished")
            # get hmV,h
            self.HmV = tor.add(self.lastHV, self.HmV)
            self.HcV = tor.add(self.lastCV, self.HcV)
            self.lastHV = self.HmV
            self.lastCV = self.HcV
            self.HmP = tor.add(self.lastHP, self.HmP)
            self.HcP = tor.add(self.lastCP, self.HcP)
            self.lastHP = self.HmP
            self.lastCP = self.HcP

            # Hs 8*1*8  d1*1*b2
            ## d3==b2  b5==b3
            # Om b2*1*numberlabel
            Hs = tor.add(self.HmP, self.HmV)
            Os = self.linear3(Hs)
            # tmp numberlabel*1 *b2
            tmp = Os.permute(2, 1, 0)
            tmp1 = self.linear4(tmp)
            # tmp1 numberlabel*1 *1
            tmp1 = tmp1.permute(2, 1, 0)
            # pro numberlabel
            probability = tmp1[0][0]
            ## feed sotfmax
            probability = probability.softmax(0)
            #Po N*numberlabel
            Po[i]=probability
            del Hs, Os, tmp, tmp1
            print("clip ", i, " finished")
        ytrue=self.initLabelMatrix(label)
        loss=self.criterion(probability,ytrue)
        loss.backward()

        return Po,loss

    def lossFunction(self, probabilities, label):
        ytrue = self.initLabelMatrix(label)
        sum = tor.sum(ytrue.mul(probabilities.log()), -1)
        [a, b] = sum.size()
        Mx = self.Mmatrix(a)
        ls = -tor.sum(tor.sum(sum.mul(Mx), -1), 0)
        return ls
    def runOnGPU(self):
        device = tor.device("cuda:0" if tor.cuda.is_available() else "cpu")
    def Mmatrix(self,M):
        mm=tor.FloatTensor(M)
        for m in range(M):
            mm[m]=m/M
        return mm

class sr_tsl(nn.Module):
    def __init__(self,scLstmDim = 8,label_number=5,is3D=True):
        super(sr_tsl,self).__init__()
        self.k = 67
        self.grnndim = 8
        self.scLstmDim = scLstmDim
        self.hiddenGRNN =8
        self.hidden3 = 8
        self.hidden4 = self.hiddenGRNN
        self.clipM = 16
        if is3D:
            self.dim = 4
        else:
            self.dim = 3


        self.d1 = self.scLstmDim
        self.d2 = 8
        self.d3 = self.d1
        self.NumberLabel = label_number
        self.ep = 5
        self.drop=0.5

        self.weight = tor.ones(self.k, self.grnndim )
        self.bias = tor.zeros(self.k, self.grnndim )
        self.lastHV = tor.zeros(self.d1, 1, self.scLstmDim)
        self.lastCV = tor.zeros(self.d1, 1, self.scLstmDim)
        self.HmV = tor.zeros(self.d1, 1, self.scLstmDim)
        self.HcV = tor.zeros(self.d1, 1, self.scLstmDim)

        self.lastHP = tor.zeros(self.d1, 1, self.scLstmDim)
        self.lastCP = tor.zeros(self.d1, 1, self.scLstmDim)
        self.HmP = tor.zeros(self.d1, 1, self.scLstmDim)
        self.HcP = tor.zeros(self.d1, 1, self.scLstmDim)

        self.lastq = tor.zeros(self.k, self.scLstmDim)
        self.netinit()




    def netinit(self):


        ##net init
        # fc 1
        self.linear = nn.Linear(self.dim, self.grnndim )
        # grnn
        self. net = grnn(weights=self.weight, bias=self.bias,input_dim=self.grnndim,hidden_dim=self.hiddenGRNN, epoch=self.ep)
        # fc2
        self.linear2 = nn.Linear(self.grnndim, self.scLstmDim)
        # skip clip
        self.skLSTM_Vec = tsl.skip_clipLSTM(1, self.scLstmDim, self.hiddenGRNN, self.d1, self.hidden3, self.d2, self.hidden4, self.d3)
        self.skLSTM_Pos = tsl.skip_clipLSTM(1, self.scLstmDim, self.hiddenGRNN, self.d1, self.hidden3, self.d2, self.hidden4, self.d3)



    def setLabelNumber(self,n=5):
        self.NumberLabel = n
    def setDevice(self,dev):
        self.lastCP.to(dev)
        self.lastHP.to(dev)
        self.HmV.to(dev)
        self.HcV.to(dev)
        self.lastHP.to(dev)
        self.lastCP.to(dev)
        self.HmP.to(dev)
        self.HcP.to(dev)
        self.lastq.to(dev)
        self.skLSTM_Vec.toDevice(dev)
        self.skLSTM_Pos.toDevice(dev)

    def forward(self,videodata):
        n = videodata.size()
        if n[0] <= 0:
            return tor.zeros(1, self.NumberLabel)
        N = math.floor(n[0] / self.clipM)

        HP=tor.FloatTensor(N,self.d1,1, self.scLstmDim)
        HV = tor.FloatTensor(N, self.d1, 1, self.scLstmDim)
        for i in range(N):
            print("clip ", i, " start")
            # Q = tor.FloatTensor(self.k, 1, self.scLstmDim)

            for j in range(self.clipM):
                o = self.linear(videodata[i * self.clipM + j])
                # grnn
                output = self.net(o)
                ## fc 2
                ##q  k*scLstmDim
                q = self.linear2(output)

                ## q->inn k*1*scLstmDim
                ## q- lastq = vector
                innV = expendRank(1, q.add( - self.lastq))
                innP = expendRank(1, q)
                self.lastq = q
                ## skip-clip lstm for one frame
                outtV, (self.lastHV, self.lastCV) = self.skLSTM_Vec(innV, self.lastHV, self.lastCV)
                outtP, (self.lastHP, self.lastCP) = self.skLSTM_Pos(innP, self.lastHP, self.lastCP)


                # print("frame ", i*clipM+j, " finished")
            # get hmV,h
            self.HmV = tor.add(self.lastHV, self.HmV)
            self.HcV = tor.add(self.lastCV, self.HcV)
            self.lastHV = self.HmV
            self.lastCV = self.HcV
            self.HmP = tor.add(self.lastHP, self.HmP)
            self.HcP = tor.add(self.lastCP, self.HcP)
            self.lastHP = self.HmP
            self.lastCP = self.HcP

            # Hs 8*1*8  d1*1*scLstmDim
            ## d3==scLstmDim  hidden4==hiddenGRNN

            #HP,HV  N* self.d1* 1*self.scLstmDim
            HP[i]=self.HmP
            HV[i]=self.HmV

            print("clip ", i, " finished")
        return HP,HV

    def extractFeature(self,videodata):
        n = videodata.size()
        if n[0] <= 0:
            return tor.zeros(1, self.NumberLabel)
        N = math.floor(n[0] / self.clipM)

        HP = tor.FloatTensor(N, self.d1, 1, self.scLstmDim)
        HV = tor.FloatTensor(N, self.d1, 1, self.scLstmDim)
        for i in range(N):
            print("clip ", i, " start")
            # Q = tor.FloatTensor(self.k, 1, self.scLstmDim)

            for j in range(self.clipM):
                o = self.linear(videodata[i * self.clipM + j])
                # grnn
                output = self.net(o)
                ## fc 2
                ##q  k*scLstmDim
                q = self.linear2(output)

                ## q->inn k*1*scLstmDim
                ## q- lastq = vector
                innV = expendRank(1, q.add(- self.lastq))
                innP = expendRank(1, q)
                self.lastq = q
                ## skip-clip lstm for one frame
                outtV, (self.lastHV, self.lastCV) = self.skLSTM_Vec(innV, self.lastHV, self.lastCV)
                outtP, (self.lastHP, self.lastCP) = self.skLSTM_Pos(innP, self.lastHP, self.lastCP)


                # print("frame ", i*clipM+j, " finished")
            # get hmV,h
            self.HmV = tor.add(self.lastHV, self.HmV)
            self.HcV = tor.add(self.lastCV, self.HcV)
            self.lastHV = self.HmV
            self.lastCV = self.HcV
            self.HmP = tor.add(self.lastHP, self.HmP)
            self.HcP = tor.add(self.lastCP, self.HcP)
            self.lastHP = self.HmP
            self.lastCP = self.HcP

            # Hs 8*1*8  d1*1*scLstmDim
            ## d3==scLstmDim  hidden4==hiddenGRNN

            # HP,HV  N* self.d1* 1*self.scLstmDim
            HP[i] = self.HmP
            HV[i] = self.HmV

            print("clip ", i, " finished")
        return tor.add(HP, HV)

class myLoss(nn.Module):



    def __init__(self):
        super(myLoss,self).__init__()
        self.device = tor.device("cpu")

    def setDevice(self,dev):
            self.device=dev

    def forward(self,input,ytrue):
        return  self.lossFunction(input,ytrue)

    def Mmatrix(self, M):
        mm = tor.FloatTensor(M)
        for m in range(M):
            mm[m] = m / M
        return mm

    def lossFunction(self, probabilities, ytrue):

        sum = tor.sum(ytrue.mul(probabilities.log()), -1)
        size = sum.size()
        if size[0]<=0:
            return 0
        Mx = self.Mmatrix(size[0])
        Mx=Mx.to(self.device)
        ls = -tor.sum(tor.sum(sum.mul(Mx), -1), 0)
        return ls



class grnn(nn.Module):
    def __init__(self,weights,bias,input_dim=512,hidden_dim=256,epoch=1):
        super(grnn,self).__init__()
        self.weights=weights
        self.bias=bias
        self.epoch=epoch
        self.relu=nn.ReLU(inplace=True)
        self.lstmcell=nn.LSTMCell(input_dim,hidden_dim)

    def forward(self,Ek):

        R0=Ek;
        [k,b]=Ek.size()
        lastR = R0.view(k, b);
        lastS=tor.zeros(k,b)

        for i in range(self.epoch):
            #print("round ",i)
            M = self.updateMessage(weight=self.weights,s=lastS,bias=self.bias)
            newS=self.updateState(r=lastR,m=M,lastS=lastS)
            newR= self.updateRelation(lastR,lastS)

            lastR=newR
            lastS=newS
        q=lastR

        return  self.relu(q)






    def updateMessage(self,weight,s,bias):
        [k,b]=s.size()
        M=tor.zeros(k,b)
        sum = tor.zeros(b)
        for j in range(k):
            sum = tor.add(sum, tor.add(tor.mul(weight[j], s[j]), bias[j]))
        for i in range(k):
            M[i]=tor.add(sum, -tor.add(tor.mul(weight[i], s[i]), bias[i]));


        return M

    def updateState(self,r,m,lastS):
        [k,b]=lastS.size()
        newS=tor.zeros(k,b)

        tr=expendRank(1,r)
        tm=expendRank(1,m)
        ts=expendRank(1,lastS)


        for i in range(k):
             pre=(Variable(tm[i]),Variable(ts[i]))
             h_state, c_state =self.lstmcell(Variable(tr[i]), pre)
             newS[i]=h_state

        return newS


    def updateRelation(self,lastR,lastS):

        return  tor.add(lastR,lastS)