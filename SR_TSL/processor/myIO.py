from SR_TSL.utils import OpenPoseReader
import  torch as tor
import numpy as np

def load_data():
        a=1
        train_data=1
        train_labels=1
        test_data=1
        test_labels=1
        return (train_data, train_labels), (test_data, test_labels)
def readOp3d(path,pattern='*ts.json'):
        js = OpenPoseReader.json_pack(path, is3D=True,pattern=pattern)
        #pose of people 0 in  frame 0

        #print(len(dt0))
        #print(dt0)
        sum=[]
        for frame in js['data']:
            if(len(frame['skeleton'])>0):
               sum.append(frame['skeleton'][0]['pose'])

        data = tor.FloatTensor(np.array(sum))
        if len(sum)<=0:
                data=tor.zeros(1,1,1)

        return  data

def readOp2d(path,pattern='*.json'):

        js = OpenPoseReader.json_pack(path,480,720, is3D=False,pattern=pattern)

        #print(len(dt0))
        #print(dt0)
        sum = []
        for frame in js['data']:
            if (len(frame['skeleton']) > 0):
                sum.append(frame['skeleton'][0]['pose'])

        data = tor.FloatTensor(np.array(sum))
        return  data