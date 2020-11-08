import argparse

configparser = argparser.ArgumentParser()
configparser.add_argument('-ne','--num_epochs',type=int,default=1000,help='Number of Epochs')
configparser.add_argument('-bs','--batch_size',type=int,default=10,  help='Batch size')
configparser.add_argument('-sl','--sequence_length',type=int,default=64,help='Sequence length')
configparser.add_argument('-lr''--learning_rate',type=float,default=2.,help='learning rate')
configparser.add_argument('-hs','--hidden_size',type=int,default=100,help='Size of the hidden vector')
configparser.add_argument('-nrl','--num_rlayers',type=int,default=1,help='Number of recurrent layers')