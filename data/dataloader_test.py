import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon

from utils.parser import configparser
from data.dataloader import dssmDataloader

if __name__ == '__main__':
    configs = configparser.parse_args()
    loader = dssmDataloader(configs)
    train_data, test_data = loader.load_data()
