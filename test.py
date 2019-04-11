
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

if __name__ == "__main__":






    '''
    train_path = "Data/train"
    model = Model()




    save_paths=[]
    save_paths.append("SR_TSL/model/m1.yml")
    save_paths.append("SR_TSL/model/m2.yml")
    save_paths.append("SR_TSL/model/m3.yml")
    save_paths.append("SR_TSL/model/m4.yml")
    save_paths.append("SR_TSL/model/m5.yml")
    save_paths.append("SR_TSL/model/m6.yml")
    save_paths.append("SR_TSL/model/m7.yml")
    print("load net")
    model.loadNet(save_paths)

    print("start train")


    model.train(train_path)
    print("train finish")
    model.saveNet(save_paths)
    print("save finish")
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

    #demo.test()














   












    ## output k*1*b3    h1 d3*1*b5

















