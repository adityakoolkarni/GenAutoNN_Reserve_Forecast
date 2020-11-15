from mxnet import gluon, nd
from mxnet.gluon import nn
import numpy as np


class DeepStateSpaceModel(nn.Block):
    def __init__(self,configs,ctx):
        super(DeepStateSpaceModel,self).__init__()

        self.h = [nd.zeros(shape=(configs.num_rlayers,configs.batch_size,configs.num_hidden),dtype=np.float64,ctx=ctx)]*2
        self.lstm = gluon.rnn.LSTM(configs.num_hidden,configs.num_rlayers,input_size=3,dtype=np.float64)
        self.linear = nn.Dense(configs.num_hidden*2+4,dtype=np.float64)

        ##scaling parameters within range
          


    def forward(self,cov_x):
        output, self.h = self.lstm(cov_x,self.h)
        ##TODO: Can we avoid swapping? 
        output = self.linear(nd.swapaxes(output,0,1))
        return output
