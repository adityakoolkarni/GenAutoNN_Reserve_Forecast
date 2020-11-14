import argparse

configparser = argparse.ArgumentParser()
configparser.add_argument('--num_epochs',type=int,default=1000,help='Number of Epochs')
configparser.add_argument('--batch_size',type=int,default=10,  help='Batch size')
configparser.add_argument('--learning_rate',type=float,default=2.,help='learning rate')
configparser.add_argument('--num_hidden',type=int,default=12+7+24,help='Size of the hidden vector')
configparser.add_argument('--num_rlayers',type=int,default=1,help='Number of recurrent layers')
configparser.add_argument('--num_years_train',type=int,default=2,help='Number of years training data')
configparser.add_argument('--csv_data_path',type=str,default='data/CAISO-20170701-20201030.csv',help='The csv path for data')
configparser.add_argument('--h5py_data_path',type=str,default='data/train_data_seq_len_7.h5',help='The h5py path for data')
configparser.add_argument('--seq_len',type=int,default=7,help='Sequence length')
