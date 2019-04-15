
import  numpy as np

import  torch as tor
from torch import  nn
from torch.autograd import Variable
from  SR_TSL.net import sr
from SR_TSL.processor import  myIO
from  SR_TSL.net import tsl
from SR_TSL.utils.tool import expendRank
from  torch.nn.functional import softmax
from SR_TSL.net.sr import Model
import math
from pathlib import Path
from SR_TSL.processor.myprocessor import myprocessor
from torch.optim import Adam
import random
from  SR_TSL.processor import demo
from  ST_GCN.processor import stprocessor
from ST_GCN.processor import STIO
if __name__ == "__main__":






    '''
    
    
    train_path = "Data/train"
    processor = myprocessor(learn=0.01)
    processor.runOnGPU(True)

    print("train start")
    processor.train(train_path,load=True,loadpath="model/test.pkl",epoch=10)
    print("finish")
    processor.saveModel()
    print("saved")
    processor.test(train_path)
    print("test finished")
    file = Path("Data/train/two/two")
    data = myIO.readOp3d(file, pattern="*ts_1.json")
    prediction=processor.predict(data)
    print("expect two prediction:",prediction)
   '''
    train_path = "Data/train"
    test_path="Data/test"
    save_path="model/test3d.yml"
    mode=0 #0 for body+hand , 1 for body
    processor= stprocessor.STprocessor()
    processor.set3D(True)
    processor.setMode(mode)
    processor.learningRate=0.005
    processor.init2()
    processor.gpu()
    processor.loadfrom(save_path)
    pattern1="*ts.json"
    pattern2 = "*ts_1.json"
    processor.train2(train_path=train_path,pattern=pattern1,epoch=16)
    processor.train2(train_path=train_path,pattern=pattern2,epoch=16)
    processor.saveTo(save_path)
    print("test start")
    processor.test(train_path,pattern=pattern1)


    print("done")











   












    ## output k*1*b3    h1 d3*1*b5

















