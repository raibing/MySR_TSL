import  torch as tor
from  torch import  optim
from  torch import  nn
from  ST_GCN.net import  st_gcn
from pathlib import Path
from ST_GCN.processor import STIO
from ST_GCN.processor import gpu
import time
import  random,math
class STprocessor():
    def __init__(self):
        self.init_environment()

    def init2(self):
        self.initmodel()
        self.load_weights()
        self.loadOptimizer()
    def init_environment(self):
        self.NumberLabel=5
        self.learningRate=0.05
        self.in_channels=4 #4 for 3d , 3 for 2d
        self.label_name=["not","tik","two","work","xx"]
        self.dev = "cpu"
        self.setlabels()
        self.is3D=True
        self.mode=0


    def set3D(self,is3D=True):
        if is3D:
            self.in_channels = 4
        else:
            self.in_channels = 3
        self.is3D=is3D
    def setMode(self,m=0):
        self.mode=m


    def gpu(self):
        if tor.cuda.is_available() :
            gpus = gpu.visible_gpu(self.arg.device)
            gpu.occupy_gpu(gpus)
            self.gpus = gpus
            self.dev = "cuda:0"
        self.model.to(self.dev)

    def initmodel(self):
        str='openpose2'
        if self.mode==1:
            str='openpose1'
        graphs = {'layout': str,
                  'strategy': 'uniform',
                  'max_hop': 1,
                  'dilation': 1}
        self.model = st_gcn.Model(in_channels=self.in_channels,num_class=self.NumberLabel,graph_args=graphs,edge_importance_weighting=False)
        # 2d , 5 classes
        self.lossfunction = nn.CrossEntropyLoss()

    def loadOptimizer(self):
        self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learningRate,
                )

    def load_weights(self):
        a=1
    def loadfrom(self,path="model/st.yml"):
        state=tor.load(path)
        self.model.load_state_dict(state)
        self.model.eval()
    def saveTo(self,path="model/st.yml"):
        tor.save(self.model.state_dict(),path)
    def  feed(self,data,label):
        # label=label_name[math.floor(random.random()*20)%3]



        output = self.model(data)
        # output=tor.rand(1,3,requires_grad=True)
        #output  1*numberlabel

        loss = self.lossfunction(output, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return  loss.item()
    def setlabels(self):
        self.Labels={}
        c=0
        for label in self.label_name :
            self.Labels[label]=c
            c+=1
    def getlabel(self,labelname):
        label=self.Labels[labelname]
        la=tor.tensor([label]).long()

        return la

    def train(self, train_path="Data/train", test_path="Data/test", load=False, loadpath="model/st.yml", is3D=True,
              pattern="*ts.json",epoch=1):
        files = Path(train_path)

        totalloss=0
        starttime = time.time()
        print("train start at",time.asctime( time.localtime(starttime) ))


        for e in range(epoch):
            print("epoch ", e)

            for Label in files.glob('*'):
                print("label name:", Label.name)
                maxsample = 12
                for sample in Label.glob('*'):
                    print("smaples name:", sample.name)
                    dirs+=(sample,Label.name)
                    if maxsample<0:
                        break
                    maxsample-=1
                    if self.is3D:
                        pose = STIO.readOp3d(sample, pattern,mode=self.mode)
                    else:
                        pose = STIO.readOp2d(sample, pattern,mode=self.mode)

                    data=tor.from_numpy(pose)
                    data= data.unsqueeze(0).float().detach()
                    data = data.to(self.dev)
                    label = self.getlabel(Label.name)
                    label=label.to(self.dev)
                    loss=self.feed(data,label)
                    totalloss+=loss
                    print("current loss",loss," at epoch ",e)
        print("total loss",totalloss)
        finished=time.time()
        print("train finished at ",time.asctime( time.localtime(finished) ))
    def train2(self, train_path="Data/train", test_path="Data/test", load=False, loadpath="model/st.yml", is3D=True,
              pattern="*ts.json",epoch=1):
        files = Path(train_path)

        totalloss=0
        starttime = time.time()
        print("train start at",time.asctime( time.localtime(starttime) ))

        dirs=[]


        for Label in files.glob('*'):
            print("label name:", Label.name)

            for sample in Label.glob('*'):
                print("smaples name:", sample.name)
                dirs.append([sample,Label.name])

        for e in range(epoch):
            print("epoch ", e)
            dirs = self.mixData(dirs)
            currentloss=0
            for sample,labelname in dirs:
                if self.is3D:
                    pose = STIO.readOp3d(sample, pattern,mode=self.mode)
                else:
                    pose = STIO.readOp2d(sample, pattern,mode=self.mode)
                data=tor.from_numpy(pose)
                data= data.unsqueeze(0).float().detach()
                data = data.to(self.dev)
                label = self.getlabel(labelname)
                label=label.to(self.dev)
                loss=self.feed(data,label)
                currentloss+=loss
            print("current loss",currentloss," at epoch ",e)
            totalloss+=currentloss
        print("total loss",totalloss)
        finished=time.time()
        print("train finished at ",time.asctime( time.localtime(finished) ))
    def test(self,test_path="Data/test",is3D=True,pattern='*ts_1.json'):

        files = Path(test_path)

        correct=0
        total=0.001
        starttime = time.time()
        print("test start at", time.asctime(time.localtime(starttime)))
        with tor.no_grad():
            for Label in files.glob('*'):
                #print("label name:", Label.name)


                for sample in Label.glob('*'):
                    if self.is3D:
                        pose = STIO.readOp3d(sample,pattern,mode=self.mode)
                    else:
                        pose=STIO.readOp2d(sample,pattern,mode=self.mode)
                    data = tor.from_numpy(pose)
                    data = data.unsqueeze(0).float().detach()
                    data = data.to(self.dev)

                    self.model.eval()
                    output,feature = self.model.extract_feature(data)
                    output = output[0]

                    prediction = output.sum(dim=3).sum(dim=2).sum(dim=1)
                    #prediction=output
                   # print(prediction)
                    prediction=prediction.argmax(dim=0)
                    print(prediction)
                    la=self.label_name[prediction]
                    print('Prediction result: {}'.format(la)," expect ",Label.name)
                    total+=1
                    if la==Label.name:
                        correct+=1
        finished = time.time()
        print("train finished at ", time.asctime(time.localtime(finished)))
        print("test correct rate:",correct/total)

    def mixData(self,datas):
        l = len(datas)
        for i in range(l):
            t=datas[i]
            x=math.floor(random.random()*131)%l
            datas[i]=datas[x]
            datas[x]=t
        return datas