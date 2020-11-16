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
from data.dataloader import dssmDataloader 
from models.dssm import DeepStateSpaceModel
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

def train_old(model,configs,dataloader,ctx):
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

def train(model,train_data,metadata,ctx):
    '''
        Get a probabilistic estimator and then train a DSSM model with data,
        save the model


    '''
    estimator = model.get_estimator(metadata)

    ## Get a predictor by training the estimator 
    tic = time.time()
    predictor = estimator.train(train_data)
    print("**"*30)
    print("Training completed sucessfully and took %s s",time.time()-toc )
    model.save_model(predictor)
    print("Saving models ")
    print("**"*30)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100)  # number of sample paths we want for evaluation

    forecasts = list(forecast_it)
    tss = list(ts_it)
    forecast_entry = forecasts[0]

    print(f"Number of sample paths: {forecast_entry.num_samples}")
    print(f"Dimension of samples: {forecast_entry.samples.shape}")
    print(f"Start date of the forecast window: {forecast_entry.start_date}")
    print(f"Frequency of the time series: {forecast_entry.freq}")

    
    

if __name__ == '__main__':
    mx.random.seed(1)
    ctx = mx.gpu(0)
    #ctx = mx.cpu()
    configs = configparser.parse_args()
    pprint(vars(configs))

    ## Getting Data ##
    #train_dataloader = CaisoDataloader(configs).make_sequence_data()
    loader = dssmDataloader(configs)
    train_data, test_data, metadata = loader.load_data()

    ## Defining the model from scratch##
    model = DeepStateSpaceModel(configs,ctx)
    #model.initialize(ctx=ctx)

    train(model,train_data,metadata,ctx)

    #train(model,configs,train_dataloader,ctx)

