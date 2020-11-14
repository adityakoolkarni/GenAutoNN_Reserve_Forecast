from mxnet import gluon, nd
from mxnet.gluon import nn
import numpy as np


class DeepStateSpaceModel(nn.Block):
    def __init__(self,configs,ctx):
        super(DeepStateSpaceModel,self).__init__()
        self.lstm = gluon.rnn.LSTM(configs.num_hidden,configs.num_rlayers,dtype=np.float64)
        self.net = nn.Sequential()
        self.net.add(gluon.rnn.LSTM(configs.num_hidden,configs.num_rlayers,dtype=np.float64),nn.Dense(configs.num_hidden*2+4,dtype=np.float64))
        
        #self.h0, self.c0 = 
        self.h = [nd.zeros(shape=(configs.num_rlayers,configs.batch_size,configs.num_hidden),dtype=np.float64,ctx=ctx)]*2

        ##TODO: Map output to model parameters

    def forward(self,cov_x):
        #output, self.h = self.lstm(cov_x,self.h)
        output = self.net(cov_x)
        return output
