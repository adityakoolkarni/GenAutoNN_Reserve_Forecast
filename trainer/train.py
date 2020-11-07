from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
mx.random.seed(1)
ctx = mx.gpu(0)

if __name__ == '__main__':
    epochs = 2000
    moving_loss = 0.
    
    learning_rate = 2.0
    
    # state = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
    for e in range(epochs):
        ############################
        # Attenuate the learning rate by a factor of 2 every 100 epochs.
        ############################
        if ((e+1) % 100 == 0):
            learning_rate = learning_rate / 2.0
        h = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
        c = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
        for i in range(num_batches):
            data_one_hot = train_data[i]
            label_one_hot = train_label[i]
            with autograd.record():
                outputs, h, c = lstm_rnn(data_one_hot, h, c)
                loss = average_ce_loss(outputs, label_one_hot)
                loss.backward()
            SGD(params, learning_rate)
    
            ##########################
            #  Keep a moving average of the losses
            ##########################
            if (i == 0) and (e == 0):
                moving_loss = nd.mean(loss).asscalar()
            else:
                moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
    
        print("Epoch %s. Loss: %s" % (e, moving_loss))
        print(sample("The Time Ma", 1024, temperature=.1))
        print(sample("The Medical Man rose, came to the lamp,", 1024, temperature=.1))
