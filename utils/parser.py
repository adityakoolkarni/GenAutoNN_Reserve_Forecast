import argparse

configparser = argparse.ArgumentParser()

configparser.add_argument('--num_epochs',
                          type=int,
                          default=1,
                          help='Number of epochs.')
configparser.add_argument('--batch_size',
                          type=int,
                          default=1000,
                          help='Batch size.')
configparser.add_argument('--learning_rate',
                          type=float,
                          default=4e-2,
                          help='Learning rate.')
configparser.add_argument('--num_hidden',
                          type=int,
                          default=12+7+24,
                          help='Dimension of latent state vector.')
configparser.add_argument('--num_layers',
                          type=int,
                          default=1,
                          help='Number of recurrent layers.')
configparser.add_argument('--data_path',
                          type=str,
                          default='data/dataset_v03.csv',
                          help='The path to the .csv data file.')
configparser.add_argument('--six_ramps',
                          type=int,
                          default=1,
                          help='Predict six component ramps vs one ramp.')
configparser.add_argument('--run_eval_only',
                          type=bool,
                          default=False,
                          help='Runs only evaluation metrics.')
configparser.add_argument('--simulation_num',
                          type=int,
                          default=1,
                          help='This is model that will be used to run evaluations on')
configparser.add_argument('--freq',
                          type=str,
                          default='1H',
                          help='Frequency of data available.')
configparser.add_argument('--model_save_path',
                          type=str,
                          default='saved_models/',
                          help='Save model here in this path.')
configparser.add_argument('--context_len',
                          type=int,
                          default=24,
                          help='Context length.')
configparser.add_argument('--pred_len',
                          type=int,
                          default=24,
                          help='Prediction length.')
