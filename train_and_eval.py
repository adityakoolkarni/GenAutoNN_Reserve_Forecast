import os
import time
import json
import datetime
import mxnet as mx
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
import numpy as np

from pathlib import Path
from utils.parser import configparser
from utils.forecast import ForecastMe
from data.dataloader import dssmDataloader
from models.dssm import DeepStateSpaceModel
from itertools import tee
import shutil
import matplotlib.pyplot as plt
import seaborn as sns


def train(model, train_ds, metadata):
    """
    Get a probabilistic estimator, train this to produce a predictor, and then
    save the predictor.
    """
    estimator = model.get_estimator(metadata)

    tic = time.time()
    predictor = estimator.train(
        training_data=train_ds
        )

    print("*"*64)
    training_time = int(time.time() - tic)
    print(f"Training took {training_time} seconds.")
    model.save_model(predictor)
    print("Predictor saved.")
    print("*"*64)


def plot_forecasts(forecast, agg_metrics, series, name, configs, err_type):
    context_len, pred_len = configs.context_len, configs.pred_len
    plot_length =  pred_len +context_len
    prediction_intervals = (50.0, 90.0)
    legend = ['observations', 'median prediction']
    legend += [f'{k} prediction interval' for k in prediction_intervals][::-1]

    plot_path = os.path.join(configs.model_save_path, 'plots')

    # first plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7),dpi=200)
    series[-plot_length:].plot(ax=ax)  # plot the time series
    forecast.plot(prediction_intervals=prediction_intervals, color='r')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    ax.set_title(name.replace('_', ' ').title())
    plt.savefig(os.path.join(plot_path, name + ".png"))

    # second plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(
        series.index[-configs.pred_len:],
        agg_metrics[err_type] 
        )
    ax.set_title(f"Percentage Error Plot for {name}")
    plt.savefig(os.path.join(plot_path, name + '_error_perc.png'))


def log_eval(configs, agg_metrics):
    # filename = datetime.datetime.now().isoformat(timespec='seconds')
    # log_content = {
    #     'configuration': vars(configs),
    #     'results': agg_metrics
    #     }
    with open(os.path.join(configs.model_save_path, 'logs.txt'), 'w') as f:
        f.writelines(['Configuration Settings ', '\n'])
        for k, v in vars(configs).items():
            f.writelines([str(k), '\t', str(v), '\n'])
    with open(os.path.join(configs.model_save_path, 'logs.txt'), 'a') as f:
        f.writelines(['Evaluation Metrics ', '\n'])
        for k, v in agg_metrics.items():
            f.writelines([str(k), '\t', str(v), '\n'])


def combine_forecast_components(forecasts_ls, series_ls):
    caiso_net_load_ramp = (
        forecasts_ls[0].samples -
        forecasts_ls[2].samples -
        forecasts_ls[4].samples
        )
    eia_net_load_ramp = (
        forecasts_ls[1].samples -
        forecasts_ls[3].samples -
        forecasts_ls[5].samples
        )

    combined_forecast_samples = { 
            "caiso_net_load_ramp_from_component_ramps": caiso_net_load_ramp,
            "eia_net_load_ramp_from_component_ramps": eia_net_load_ramp
            }

    return combined_forecast_samples

def error_percentage(predicted,true):
    '''
    Computes error percentage and plots the same 
    params:
        predicted - num_prediction_paths x prediction_len
        true - 1 x prediction_len
    '''
    err_perc = (np.mean(predicted,axis=0) - true.reshape(-1)) * 100 / true.reshape(-1)
    return err_perc 

def eval(model, validation_ds, metadata, configs, names,
         caiso_net_load_ramp=None, eia_net_load_ramp=None):
    predictor = model.load_model()

    forecasts_it, series_it = make_evaluation_predictions(
        dataset=validation_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=128  # number of sample paths we want for evaluation
        )

    forecasts_list_it, forecast_it = tee(forecasts_it)
    series_list_it, series_it = tee(series_it)
    forecasts_ls = list(forecasts_list_it)
    series_ls = list(series_list_it)

    evaluator = Evaluator(quantiles=[0.10, 0.50, 0.90])
    agg_metrics, item_metrics = evaluator(
        series_it,
        forecast_it,
        num_series=len(validation_ds)
        )

    for i in range(len(forecasts_ls)):
        forecast = forecasts_ls[i]
        name = names[i].lower()
        print("*-"*40)
        print("*"*40, end='')
        print(f"Forecast {name}", "*"*30)
        print(f"Number of sample paths: {forecast.num_samples}")
        print(f"Dimension of samples: {forecast.samples.shape}")
        print(f"Start date of the forecast window: {forecast.start_date}")
        print(f"Frequency of the time series: {forecast.freq}")
        print("*-"*40)
        series = series_ls[i]
        err_type = name + '_error_perc'
        agg_metrics[err_type] = error_percentage(predicted=forecast.samples,
                                                            true=series[-configs.pred_len:].to_numpy())
        plot_forecasts(forecast, agg_metrics, series, name, configs, err_type)

    # now do a similar calc for the combined forecasts
    # log these values - I have logged the error percentages - Aditya
    plt.close('all')
    if configs.six_ramps:
        target_ramp_caiso = series_ls[0] - series_ls[2] - series_ls[4] 
        target_ramp_eia   = series_ls[1] - series_ls[3] - series_ls[5] 
        combined_forecast_samples = combine_forecast_components(
            forecasts_ls,
            series_ls
            )

        for cur_label,cur_samples in combined_forecast_samples.items():
            if "caiso" in cur_label:
                cur_target = caiso_net_load_ramp
                err_type = 'caiso_error_perc'
                ramp_name = 'CAISO Ramp Prediction Mean' if 'mean' in cur_label else 'CAISO Ramp Prediction Median'
                target_series = target_ramp_caiso
            else:
                cur_target = eia_net_load_ramp
                err_type = 'eia_error_perc'
                ramp_name = 'EIA Ramp Prediction Mean' if 'mean' in cur_label else 'EIA Ramp Prediction Median'
                target_series = target_ramp_eia
            agg_metrics[err_type] = error_percentage(predicted=cur_samples,
                                                    true=cur_target[-configs.pred_len:].reshape(-1))

            #samples: np.ndarray, start_date: pd.Timestamp, freq: 'H'):
            final_forecasts = ForecastMe(samples=cur_samples,start_date=forecast.start_date,freq='H') 
            plot_forecasts(final_forecasts, agg_metrics, series, cur_label, configs, err_type)
            #plot_final_ramps(cur_samples, target_series, ramp_name, configs)
            print(f"{cur_label} percent error: {agg_metrics[err_type]}")

    print('\nTarget series forecast plots saved.')
    log_eval(configs, agg_metrics)
    shutil.make_archive(os.path.join(configs.model_save_path,'plots'),'zip',base_dir=os.path.join(configs.model_save_path,'plots'))
    print('\nConfiguration and results logged.')


if __name__ == '__main__':
    plt.close('all')
    mx.random.seed(1)
    configs = configparser.parse_args()
    ctx = mx.gpu() if configs.use_gpu else mx.cpu()
    sim_number = len(os.listdir('./saved_models'))
    if configs.run_eval_only:
        #if 'simulation_num_' + str(configs.simulation_num) not in os.listdir('./saved_models'):
        if configs.simulation_folder_name not in os.listdir('./saved_models'):
            raise ValueError('This simulation has not been trained')
        sim_number = configs.simulation_num
        configs.model_save_path +=  configs.simulation_folder_name
        configs.model_save_path = Path(configs.model_save_path)
    else:
        configs.model_save_path += f'/simulation_num_{sim_number}'
        configs.model_save_path = Path(configs.model_save_path)
        if not os.path.isdir(configs.model_save_path):
            os.mkdir(configs.model_save_path)

        plot_path = os.path.join(configs.model_save_path, 'plots')
        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)


    loader = dssmDataloader(configs)
    # ratios of train and validation data, with the remainder as the test set
    splits = [0.80, 0.10]
    (
        clipped_datasets,
        full_datasets,
        metadatas,
        target_names,
        caiso_net_load_ramp,
        eia_net_load_ramp
    ) = loader.load_data(splits)


    print('*-'*40)
    print('*-'*40)
    print("Data Loading Complete")
    # make this very clear...
    train_ds = clipped_datasets[0]
    train_ds_full = full_datasets[0]
    train_metadata = metadatas[0]
    validation_ds = full_datasets[1]
    validation_metadata = metadatas[1]
    test_ds = full_datasets[2]
    test_metadata = metadatas[2]

    # take a look at what an entry of each ListDataset object looks like
    train_ds_entry = next(iter(train_ds))
    validation_ds_entry = next(iter(validation_ds))
    print('\n')
    print('Shape of train_ds_entry target:')
    print(train_ds_entry['target'].shape)
    print("Train_ds metadata:")
    print(json.dumps(train_metadata, indent=4))
    print('Shape of validation_ds_entry target:')
    print(validation_ds_entry['target'].shape)
    print("validation_ds metadata:")
    print(json.dumps(validation_metadata, indent=4))
    print('\n')

    print('*-'*40)
    print('*-'*40)
    print("Data Loading Complete")
    configs.train_len = train_ds_entry['target'].shape[0]
    pprint(vars(configs))
    model = DeepStateSpaceModel(configs, ctx)

    if configs.run_eval_only:
        print('*-'*40)
        print('*-'*40)
        print("Running evaluation only")
        eval(model, train_ds_full, train_metadata, configs, target_names,
             caiso_net_load_ramp, eia_net_load_ramp)
        print('*-'*40)
        print('*-'*40)

    else:
        print('*-'*40)
        print('*-'*40)
        print("Running training")
        train(model, train_ds, train_metadata)
        print('*-'*40)

        print("Running evaluation only")
        eval(model, train_ds_full, train_metadata, configs, target_names,
         caiso_net_load_ramp, eia_net_load_ramp)
        print('*-'*40)
        print('*-'*40)
