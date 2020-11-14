from mxnet import gluon, nd
from mxnet.gluon import nn


class DeepStateSpaceModel(nn.Block):
    def __init__(self,configs):
        super(DeepStateSpaceModel,self).__init__()
        self.lstm = gluon.rnn.LSTM(configs.num_hidden,configs.num_rlayers)
        
        #self.h0, self.c0 = 
        self.h = [nd.zeros(shape=(configs.num_rlayers,configs.batch_size,configs.num_hidden))]*2
        ##TODO: Map output to model parameters

    def forward(self,cov_x):
        output, self.h = self.lstm(cov_x,self.h)
        return output,self.h
