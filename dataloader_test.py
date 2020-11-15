import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import gluon

from dataloader import dssmDataloader
from utils.parser import configparser

if __name__ == '__main__':
    configs = configparser.parse_args()
    loader = dssmDataloader(configs)
    train_data, test_data = loader.load_data()
