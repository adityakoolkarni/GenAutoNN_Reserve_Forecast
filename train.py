from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gloun
import numpy as np
from mx.gluon.rnn import LSTM

import os, sys, time
from utils.issm_loss import ISSM as LL #Log-Likelihood
from utils.parser import configparser
from utils.decode_params import get_ssm_params




def train(lstm_rnn,configs):
    moving_loss = 0.
    ############################
    # Data Loader and Optimizer
    ############################
    trainer = gloun.Trainer(lstm_rnn.collect_params(),'sgd',{'learning_rate':configs.lr})
    covariate_x_train = None

    for e in range(configs.num_epochs):
        ############################
        # Attenuate the learning rate by a factor of 2 every 100 epochs.
        ############################
        if ((e+1) % 100 == 0):
            configs.lr = configs.lr / 2.0
        h = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
        c = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
        for i in range(num_batches):
            tic = time.time()
            with autograd.record():
                outputs, h, c = lstm_rnn(covariate_x_train[i], h, c) #sequence length x batchsize x input_size,h,c 
                loss = LL(get_ssm_params(outputs))
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

    ## Defining the model ##
    model = LSTM(configs.hidden_size,configs.num_rlayers)

    train(model,configs)
