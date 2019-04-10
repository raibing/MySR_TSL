import math
import torch as tor
from pathlib import Path
from SR_TSL.processor import myIO
from torch.optim import Adam
import SR_TSL.net.sr as myNet
from SR_TSL.net.tsl import LearningClassier
import time
class myprocessor:
    def __init__(self):
        a=1
        self.sc=8
        self.NumberLabel = 6
        self.learningrate=0.0001
        self.momentum=0.9

        self.device = tor.device("cpu")
        self.init2()

    def vecInit(self):
        a=1
    def initModel(self,is3D=True):
        self.sr_tsl=myNet.sr_tsl(scLstmDim=self.sc,label_number=self.NumberLabel,is3D=is3D)
        self.learner = LearningClassier(self.sc,self.NumberLabel)
        self.optimizer = Adam(self.sr_tsl.parameters(), lr=self.learningrate)
        self.setDevice()

    def init2(self):

        self.cirterion=myNet.myLoss()

    def loadModel(self,path="model/test.pkl"):

        model = tor.load(path)
        self.learner.load_state_dict(model)
        self.learner.eval()

    def saveModel(self,path="model/test.pkl"):
        tor.save(self.learner,path)


    def fit(self,videodata,label):
        with tor.no_grad():
            HP,HV=self.sr_tsl(videodata)
            HP=HP.detach()
            HV=HV.detach()
        HS=tor.add(HP,HV)
        probabilities=self.learner(HS)
        ytrue = self.initLabelMatrix(label)
        ytrue=ytrue.to(self.device)
        loss=self.cirterion(probabilities,ytrue)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return probabilities,loss.item()


    def train(self,train_path="Data/train",test_path="Data/test",load=False,loadpath="model/test.pkl",is3D=True,epoch=1):
        files = Path(train_path)
        ee = 100
        self.setLabels(files)
        self.initModel(is3D)

        if load:
            self.loadModel(loadpath)


        ls = 0
        starttime=time.time()
        print("start at:",time.asctime( time.localtime(starttime) ))
        self.sr_tsl.zero_grad()
        for e in range(epoch):
            for label in files.glob('*'):
                print("label name:", label.name)

                if ee < 0:
                    break
                ee = ee - 1
                for sample in label.glob('*'):
                    print("smaples name:", sample.name)

                    data = myIO.readOp3d(sample)

                    data=data.to(self.device)

                    probabilities,loss = self.fit(data,label)
                    ls+=loss
                    del  data
                    print("loss:", ls)
                print("label finish at:",time.asctime( time.localtime(time.time())))
            print("epoch ",epoch," finished with loss ",ls)
        current=time.time()
        print("train finish at:",time.asctime( time.localtime(current)))
        print("cost ",current-starttime," ms")

    def test(self,test_path):

        print("test start at ",time.time())
        files = Path(test_path)
        self.setLabels(files)
        self.initModel(True)
        correct=0
        total=0.001
        pattern='*ts_1.json'
        e=100


        with tor.no_grad():
            for label in files.glob('*'):
                print("label name:", label.name)
                if e<=0:
                    break
                e-=1
                for sample in label.glob('*'):
                    data = myIO.readOp3d(sample,pattern=pattern)

                    data = data.to(self.device)
                    print(data.size())
                    HP, HV = self.sr_tsl(data)
                    HS = tor.add(HP, HV)
                    pro = self.learner(HS)
                    ##pro n*numberlabel
                    size = pro.size()
                    if size[0] <= 0:
                        return 0
                    predictions=tor.zeros(size)
                    for i in range(size[0]):
                        predictions[i][tor.argmax(pro[i])]=1
                    sum=tor.sum(predictions,0)
                    x=tor.argmax(sum)
                    pre = sum[x]/size[0]
                    correct+=(self.labels[x]==label)
                    total+=1
                    print("this sample ",label.name," is predict to be :",self.labels[x]," with confidence ",pre)
        print("label ",label.name," correct rate :",correct/total)
        print("test finished at",time.time())
    def predict(self,videodata):

        with tor.no_grad():
            HP,HV= self.sr_tsl(videodata)
            HS = tor.add(HP, HV)
            pro = self.learner(HS)
            ##pro n*numberlabel
            size=pro.size()
            if size[0]<=0:
                return "unknow"
            unknow=0
            predictions=[]
            for i in range(size[0]):
                x=tor.argmax(pro[i])
                predictions.append(self.labels[x])
            prediction = tor.zeros(size)
            for i in range(size[0]):
                prediction[i][tor.argmax(pro[i])] = 1
            sum = tor.sum(prediction, 0)
            x = tor.argmax(sum)
            pre = sum[x] / size[0]
            print("this sample is predict to be :", self.labels[x], " with confidence ", pre)
            return predictions


    def setDevice(self):
        self.cirterion.setDevice(self.device)
        self.sr_tsl.setDevice(self.device)
        self.sr_tsl.to(self.device)
        self.learner.to(self.device)

    def runOnGPU(self,onGpu):
        if onGpu:
            self.device=tor.device("cuda:0" if tor.cuda.is_available() else "cpu")
        else:
            self.device=tor.device("cpu")




    def initLabelMatrix(self,label):
        matrix=tor.zeros(self.NumberLabel)
        matrix[self.Classes[label]]=1
        return  matrix

    def setLabels(self, files):
        labels = []
        Classes = {}
        c = 1
        Classes["unknow"] = 0
        labels.append("unknow")
        for lab in files.glob('*'):
            labels.append(lab.name)
            Classes[lab] = c
            c = c + 1

        self.labels = labels
        self.NumberLabel = len(labels)
        self.Classes = Classes


