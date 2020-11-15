import argparse

configparser = argparse.ArgumentParser()

configparser.add_argument('--n_epochs',
                          type=int,
                          default=1000,
                          help='Number of epochs.')
configparser.add_argument('--batch_size',
                          type=int,
                          default=10,
                          help='Batch size.')
configparser.add_argument('--learning_rate',
                          type=float,
                          default=2.,
                          help='Learning rate.')
configparser.add_argument('--n_hidden',
                          type=int,
                          default=12+7+24,
                          help='Dimension of latent state vector.')
configparser.add_argument('--n_rlayers',
                          type=int,
                          default=1,
                          help='Number of recurrent layers.')
configparser.add_argument('--n_years_train',
                          type=int,
                          default=2,
                          help='Number of years in training data.')
configparser.add_argument('--data_path',
                          type=str,
                          default='dataset_v01.csv',
                          help='The path to the .csv data file.')
configparser.add_argument('--h5py_data_path',
                          type=str,
                          default='data/train_data_seq_len_7.h5',
                          help='The path to the h5py data.')
configparser.add_argument('--seq_len',
                          type=int,
                          default=24,
                          help='Sequence length.')
