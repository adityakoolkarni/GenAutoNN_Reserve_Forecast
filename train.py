from __future__ import print_function
from mxnet import nd, autograd, gluon
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet.gluon.rnn import LSTM
from pprint import pprint

import os, sys, time
from utils.issm_loss import ISSM as LL #Log-Likelihood
from utils.parser import configparser
from data.caiso_dataloader import CaisoDataloader
from models.dssm import DeepStateSpaceModel
#from utils.decode_params import get_ssm_params

def train(model,configs,dataloader,ctx):
    moving_loss = 0.
    ############################
    # Data Loader and Optimizer
    ############################
    trainer = gluon.Trainer(model.collect_params(),'sgd',{'learning_rate':configs.learning_rate})
    #num_batches = len(ramp_z_train) // configs.batch_size

    for e in range(configs.num_epochs):
        for cov_x,ramp_z in zip(dataloader[0],dataloader[1]):
            tic = time.time()
            with autograd.record():
                cov_x = nd.swapaxes(cov_x,0,1).copyto(ctx)
                parameters = model(cov_x) #batchsize x sequence length x input_size,h
                print(parameters.shape)
                #print(outputs.shape,h[0].shape,h[1].shape)
                exit(0)
                loss = -LL(ramp_z,get_ssm_params(outputs))
                loss.backward()
            trainer.step(batch_size=configs.batch_size)
            loss_scalar = loss.mean().asscalar()
            loss_log.append(loss_scalar)
            print('Epoch {} Loss {} Time {}'.format(e,loss_scalar,time.time()-tic))
    
            ##########################
            #  Keep a moving average of the losses
            ##########################
            if (i == 0) and (e == 0):
                moving_loss = nd.mean(loss).asscalar()
            else:
                moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
    
        print("Epoch %s. Loss: %s" % (e, moving_loss))

if __name__ == '__main__':
    mx.random.seed(1)
    ctx = mx.gpu(0)
    configs = configparser.parse_args()
    pprint(vars(configs))

    ## Defining the model ##
    model = DeepStateSpaceModel(configs,ctx)
    model.initialize(ctx=ctx)

    ## Getting Data ##
    #train_dataloader = CaisoDataloader(configs)
    train_dataloader = CaisoDataloader(configs).make_sequence_data()
    train(model,configs,train_dataloader,ctx)

    ##TODO Get Model parameters from LSTM
