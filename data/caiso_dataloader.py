import pandas as pd 
import mxnet as mx
from multiprocessing import cpu_count
import numpy as np
import h5py

class CaisoDataloader:
    def __init__(self,configs):
        self.df = pd.read_csv(configs.csv_data_path)
        self.df['start_dt'] = pd.to_datetime(self.df['start_dt'])
        self.df['end_dt'] = pd.to_datetime(self.df['end_dt'])
        self.df['covariate_x'] = self.df['end_dt'].apply(lambda i:np.array([float(i.week),float(i.dayofweek),float(i.hour)]))
        self.df['ramp_z'] = self.df['total_integrated_load_MW'] - self.df['solar_generation_MW'] - self.df['wind_generation_MW']
        self.configs = configs
        #self.train_data_loader = self.get_sequence_data(self.df,configs)
        #self.val_data_loader = mx.gluon.data.DataLoader(self.df[configs.train_size:configs.val_size],configs.batch_size,num_workers=cpu_count())
        #self.test_data_loader = mx.gluon.data.DataLoader(self.df[configs.val_size:configs.test_size],configs.batch_size,num_workers=cpu_count())
        #return self.make_sequence_data(self.df,configs)
    
    def make_sequence_data(self):
        df, configs = self.df, self.configs
        if configs.seq_len == 7 and 'seq_len_7.' in configs.h5py_data_path:
            print('*'*60)
            print('*'*60)
            with h5py.File(configs.h5py_data_path,'r') as f:
                covariate_x = f['covariate_x'][:]
                ramp_z = f['ramp_z'][:]

            print("Reading from saved data from ",configs.h5py_data_path, "there are total time points",len(ramp_z))
            print('*'*60)
            print('*'*60)
        else: 
            #raise NotImplementedError
            print('*'*60)
            print('*'*60)
            print("Creating data from csv data ",configs.h5py_data_path)
            print('*'*60)
            print('*'*60)
            covariate_x = self.chunkify(df['covariate_x'],configs.seq_len)
            ramp_z      = self.chunkify(df['ramp_z'],configs.seq_len)

        covariate_dataloader = mx.gluon.data.DataLoader(covariate_x,configs.batch_size,num_workers=cpu_count())
        ramp_dataloader = mx.gluon.data.DataLoader(ramp_z,configs.batch_size,num_workers=cpu_count())
        return covariate_dataloader,ramp_dataloader 
    
    def chunkify(self,series,seq_len):
        data = np.zeros((len(series) - seq_len,seq_len)) if series[0].values.to_numpy().ndim == 0  else  np.zeros((len(series) - seq_len,seq_len,3))
        for d in range(len(series)-seq_len-1):
            data[d] = np.array([series[d:d+seq_len].tolist()],dtype=np.float64)
        return data  
